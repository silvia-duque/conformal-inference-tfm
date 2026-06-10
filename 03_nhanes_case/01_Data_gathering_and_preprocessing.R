# ==============================================================================
# DATA INGESTION & PREPROCESSING PIPELINE (NHANES CYCLE L)
# ==============================================================================
#
# Attribution Note:
# This data pipeline, including the specific CDC NHANES Cycle L module extraction, 
# filtering criteria (dialysis and pregnancy exclusions), and clinical proxy 
# derivations, is based on the methodology and script:
#
#   "NHANES: A Super Learner Approach to Predicting Serum Creatinine" 
#   Developed by: Miguel Angel Luque Fernandez (Version: 2026-03-27).
# ==============================================================================

if (!require("pacman")) install.packages("pacman")
pacman::p_load(
    nhanesA,         # NHANES data retrieval API
    tidyverse,       # Data wrangling and visualization framework
    labelled,        # Variable labels and data dictionary handling
    knitr,           # Dynamic table generation
    kableExtra,      # Advanced table styling and formatting
    skimr            # Summary of data
)

# 1. Robust Data Acquisition Function
# Adds a 1-second delay between requests to respect CDC server limits and 
# prevent timeouts. Halts execution with a clear error message if download fails.
fetch_nhanes <- function(code) {
    Sys.sleep(1) 
    data <- nhanes(code)
    if (is.null(data)) {
        stop(paste("Failed to download NHANES module:", code, 
                   ". The server might be down or the code is incorrect."))
    }
    return(data)
}

# Retrieve Core Modules
demo_raw <- fetch_nhanes('DEMO_L')    # Demographics
bmx_raw  <- fetch_nhanes('BMX_L')     # Body measures (Anthropometry)
lab_raw  <- fetch_nhanes('BIOPRO_L')  # Biochemistry / Standard Labs
glu_raw  <- fetch_nhanes('GLU_L')     # Fasting Glucose
tri_raw  <- fetch_nhanes('TRIGLY_L')  # Triglycerides

# Retrieve Auxiliary Modules
dia_raw  <- fetch_nhanes('DIQ_L')     # Diabetes Questionnaire
bpq_raw  <- fetch_nhanes('BPQ_L')     # Hypertension Questionnaire
mcq_raw  <- fetch_nhanes('MCQ_L')     # Medical Conditions
ghb_raw  <- fetch_nhanes('GHB_L')     # Glycohemoglobin
kiq_raw  <- fetch_nhanes('KIQ_U_L')   # Kidney Conditions (Urine)
rx_raw   <- fetch_nhanes('RXQ_RX_L')  # Prescription Medications
bpx_raw  <- fetch_nhanes('BPXO_L')    # Oscillometric Blood Pressure (Cycle L Specific)

# 2. Data Merging Strategy
# We apply inner_join for core physiological data and left_join for auxiliary 
# questionnaires to preserve sample size integrity in case of missing modules.
df_full <- demo_raw %>%
    inner_join(bmx_raw,  by = "SEQN") %>%
    inner_join(lab_raw,  by = "SEQN") %>%
    inner_join(glu_raw,  by = "SEQN") %>%
    inner_join(tri_raw,  by = "SEQN") %>%
    left_join(dia_raw,   by = "SEQN") %>%
    left_join(bpq_raw,   by = "SEQN") %>%
    left_join(mcq_raw,   by = "SEQN") %>%
    left_join(ghb_raw,   by = "SEQN") %>%
    left_join(kiq_raw,   by = "SEQN") %>%
    left_join(bpx_raw,   by = "SEQN") %>%
    select(
        SEQN,
        # Sociodemographic Attributes
        RIAGENDR, RIDAGEYR, RIDRETH3, INDFMPIR, RIDEXPRG,
        # Anthropometric Measurements
        BMXWT, BMXHT, BMXBMI, BMXWAIST,
        # Laboratory Profiles (Target Outcome + Predictors)
        contains("SCR"), contains("SUA"), contains("SAL"), contains("STP"),
        matches("SCH"), matches("GLU"), matches("STR"), matches("SCA"),
        matches("SPH"), matches("ALP"), matches("ALT"), matches("AST"),
        contains("GH"),
        # Clinical Flags / Questionnaire Responses
        any_of(c("DIQ010", "DIQ050", "DIQ070", "BPQ020", "BPXOSY1", "MCQ160L", "KIQ025"))
    ) %>%
    # Enforce retention of weight metrics (mandatory for Cockcroft-Gault validations)
    filter(!is.na(BMXWT))

# Display Initial Dataset Dimensions
# print(dim(df_full))

# Data Quality Exploratory Summary
# skimr::skim(df_full)

# 3. Data Type Standardization & Alignment
# Force standard numeric parsing across key clinical attributes
df_full$KIQ025   <- as.numeric(df_full$KIQ025)
df_full$RIDEXPRG <- as.numeric(df_full$RIDEXPRG)
df_full$RIAGENDR <- as.numeric(df_full$RIAGENDR)
df_full$DIQ010   <- as.numeric(df_full$DIQ010)
df_full$BPQ020   <- as.numeric(df_full$BPQ020)
df_full$DIQ050   <- as.numeric(df_full$DIQ050)
df_full$DIQ070   <- as.numeric(df_full$DIQ070)
df_full$MCQ160L  <- as.numeric(df_full$MCQ160L)
rx_raw$RXQ033    <- as.numeric(rx_raw$RXQ033)
rx_raw$RXQ050    <- as.numeric(rx_raw$RXQ050)

# 4. Clinical Exclusion Criteria
n_raw <- nrow(df_full)

df_excl <- df_full %>%
    # CRITICAL EXCLUSION 1: Remove dialysis patients (KIQ025 == 1)
    # Dialysis artificially clears creatinine from the bloodstream. Thus, serum 
    # values do not reflect baseline metabolic renal clearance or true muscle mass.
    filter(is.na(KIQ025) | KIQ025 != 1) %>%
    
    # CRITICAL EXCLUSION 2: Remove pregnant women (RIAGENDR == 2 & RIDEXPRG == 1)
    # Pregnancy physiologically increases glomerular filtration rate (GFR) by ~50%, 
    # making serum creatinine systematically lower and clinically non-comparable.
    filter(!(RIAGENDR == 2 & !is.na(RIDEXPRG) & RIDEXPRG == 1))

# Report Sample Attrition Summary
cat(sprintf(
    "Raw N = %d | After clinical exclusions N = %d | Excluded N = %d\n",
    n_raw, nrow(df_excl), n_raw - nrow(df_excl)
))

# 5. Feature Engineering & Clinical Derivative Mapping
df_processed <- df_excl %>%
    mutate(
        # Race/ethnicity parsed as an explicit grouping factor
        # (Black race is correlated with higher mean muscle mass and baseline creatinine)
        race_f = RIDRETH3,
        
        # Consolidated Binary Clinical Diagnostic Flags
        diabetes_dx  = if_else(!is.na(DIQ010)  & DIQ010  == 1, 1L, 0L),
        htn_dx       = if_else(!is.na(BPQ020)  & BPQ020  == 1, 1L, 0L),
        ckd_self_rpt = if_else(!is.na(MCQ160L) & MCQ160L == 1, 1L, 0L),
        
        # Recode biological sex to binary indicator (1 = Female, 0 = Male)
        female = if_else(RIAGENDR == 2, 1L, 0L)
    )

# 6. Medication Proxy Processing (Cycle L Specific Mapping)
# Extract available pharmacological indicators and unify naming structures safely
rx_summary <- rx_raw %>%
    select(SEQN, any_of(c("RXQ033", "RXDCOUNT", "RXQ050"))) %>%
    rename(
        any_rx_raw = any_of("RXQ033"),
        n_rx_raw   = any_of(c("RXDCOUNT", "RXQ050"))
    ) %>%
    mutate(
        # Convert NHANES scale (1 = Yes, 2 = No) to strict 0/1 binary format
        any_rx = if_else(!is.na(any_rx_raw) & any_rx_raw == 1, 1L, 0L),
        # Impute missing medication counts to 0 instances
        n_rx   = if_else(is.na(n_rx_raw), 0L, as.integer(n_rx_raw))
    ) %>%
    select(SEQN, any_rx, n_rx)

# Join medication profiles and evaluate derivative therapeutic flags
df_processed <- df_processed %>%
    left_join(rx_summary, by = "SEQN") %>%
    mutate(
        any_rx = replace_na(any_rx, 0L),
        n_rx   = replace_na(n_rx, 0L),
        
        # Establish operational proxies using pharmacological profiles
        rx_antihtn_proxy = if_else(!is.na(htn_dx) & htn_dx == 1 & any_rx == 1, 1L, 0L),
        rx_insulin       = if_else(!is.na(DIQ050) & DIQ050 == 1, 1L, 0L),
        rx_antidiab_oral = if_else(!is.na(DIQ070) & DIQ070 == 1, 1L, 0L)
    )

cat("Medication flags successfully updated using variable RXQ050.\n")

# 7. Final Feature Matrix Selection & Missing Data Handling
df_final <- df_processed %>%
    select(
        # Target Variable (Serum Creatinine)
        LBXSCR,
        
        # Biomarkers & Laboratory Covariates
        LBXSAL, LBXSTP, LBXSUA, LBXSTR, LBXGLU,
        # Laboratory Extras
        LBXSCH, LBXSCA, LBXSPH, LBXGH,
        
        # Demographics & Anthropometrics
        RIAGENDR, RIDAGEYR, BMXBMI, BMXWAIST, RIDRETH3,
        
        # Clinical Covariates & Questionnaire Responses
        diabetes_dx, htn_dx, ckd_self_rpt, BPXOSY1,
        
        # Medication Profiles
        any_rx, n_rx, rx_antihtn_proxy, rx_insulin, rx_antidiab_oral
    ) %>%
    # Execute complete-case analysis for the conformal predictive pipeline
    drop_na()

# Print Final Modeling Cohort Size
print(dim(df_final))


write.csv(df_final, "nhanes_final.csv", row.names = FALSE)
cat("Data pipeline completed successfully. Clean dataset exported to nhanes_final.csv\n")
