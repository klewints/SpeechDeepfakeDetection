#!/bin/bash
# Speech Deepfake Detection - Complete Execution Script
# This script executes the full pipeline: train CNN → train Wav2Vec2 → evaluate both

# Color output for readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}       SPEECH DEEPFAKE DETECTION - DUAL PIPELINE EXECUTION${NC}"
echo -e "${BLUE}================================================================================${NC}\n"

# Step 1: Verify setup
echo -e "${YELLOW}[1/4] VERIFYING PROJECT SETUP...${NC}"
python verify_setup.py
if [ $? -ne 0 ]; then
    echo -e "${RED}Setup verification failed. Please fix issues above.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Setup verified${NC}\n"

# Step 2: Train CNN
echo -e "${YELLOW}[2/4] TRAINING CNN MODEL (Original Pipeline)...${NC}"
echo "Estimated time: 5-10 minutes"
python train.py
if [ $? -ne 0 ]; then
    echo -e "${RED}CNN training failed.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ CNN training complete${NC}\n"

# Step 3: Train Wav2Vec2
echo -e "${YELLOW}[3/4] TRAINING WAV2VEC2 MODEL (New Pipeline)...${NC}"
echo "Estimated time: 20-40 minutes (first run downloads model)"
python train_wav2vec.py
if [ $? -ne 0 ]; then
    echo -e "${RED}Wav2Vec2 training failed.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Wav2Vec2 training complete${NC}\n"

# Step 4: Evaluate both models
echo -e "${YELLOW}[4/4] EVALUATING BOTH MODELS...${NC}"
python evaluate_models.py
if [ $? -ne 0 ]; then
    echo -e "${RED}Evaluation failed.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Evaluation complete${NC}\n"

# Final summary
echo -e "${BLUE}================================================================================${NC}"
echo -e "${GREEN}                    ALL PIPELINE STEPS COMPLETED!${NC}"
echo -e "${BLUE}================================================================================${NC}\n"

echo -e "${YELLOW}Generated Outputs:${NC}"
echo "  ✓ outputs/model.pth"
echo "  ✓ outputs/wav2vec2_model_best.pth"
echo "  ✓ outputs/wav2vec2_model_final.pth"
echo "  ✓ outputs/confusion_matrix_CNN_Mel.png"
echo "  ✓ outputs/confusion_matrix_Wav2Vec2.png"
echo "  ✓ outputs/roc_curves_comparison.png"
echo "  ✓ outputs/evaluation_results.csv"
echo ""

echo -e "${YELLOW}Next Steps:${NC}"
echo "  1. Review evaluation_results.csv for metrics comparison"
echo "  2. View PNG plots in outputs/ directory"
echo "  3. Read IMPLEMENTATION.md for interpretation of results"
echo "  4. Check PIPELINE.md for technical details"
echo ""

echo -e "${BLUE}================================================================================${NC}"
