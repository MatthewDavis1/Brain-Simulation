#!/bin/bash

# Output directory
OUTPUT_DIR="examples/modular_3"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Simulation parameters
NUM_NEURONS=100
TIMESTEPS=100
CONNECTION_TYPE="modular"  # Options: random, small_world, modular
LEARNING_RULE="hebbian"
INIT_POTENTIAL_MEAN=50
INIT_POTENTIAL_STD=10
LEARNING_RATE=0.5
LEAK_RATE=0.95

# New parameters for driving inputs
NUM_DRIVERS=0
DRIVER_POTENTIAL=10

# Write parameters to file
echo "Writing simulation parameters..."
cat << EOF > "$OUTPUT_DIR/parameters.txt"
NUM_NEURONS=$NUM_NEURONS
TIMESTEPS=$TIMESTEPS
CONNECTION_TYPE=$CONNECTION_TYPE
LEARNING_RULE=$LEARNING_RULE
INIT_POTENTIAL_MEAN=$INIT_POTENTIAL_MEAN
INIT_POTENTIAL_STD=$INIT_POTENTIAL_STD
LEARNING_RATE=$LEARNING_RATE
LEAK_RATE=$LEAK_RATE
NUM_DRIVERS=$NUM_DRIVERS
DRIVER_POTENTIAL=$DRIVER_POTENTIAL
EOF

# Run simulation
echo "Starting neural network simulation..."
python3 simulation.py \
    --neurons $NUM_NEURONS \
    --timesteps $TIMESTEPS \
    --connection $CONNECTION_TYPE \
    --learning $LEARNING_RULE \
    --init-mean $INIT_POTENTIAL_MEAN \
    --init-std $INIT_POTENTIAL_STD \
    --learning-rate $LEARNING_RATE \
    --leak-rate $LEAK_RATE \
    --num-drivers $NUM_DRIVERS \
    --driver-potential $DRIVER_POTENTIAL \
    --output-dir "$OUTPUT_DIR" \
    --stats-file "simulation_stats.txt"

# Check if simulation was successful
if [ $? -ne 0 ]; then
    echo "Simulation failed!"
    exit 1
fi

echo "Simulation complete. Starting analysis..."

# Run analysis on the generated data
python3 analysis.py \
    "$OUTPUT_DIR/neuron_potentials.csv" \
    --output-dir "$OUTPUT_DIR"

# Check if analysis was successful
if [ $? -ne 0 ]; then
    echo "Analysis failed!"
    exit 1
fi

echo "Analysis complete. Results are available in $OUTPUT_DIR/"
echo "Generated files:"
ls -l "$OUTPUT_DIR" 