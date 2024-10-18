#!/usr/bin/env bash
main() {
# Define the current version of the script
CURRENT_VERSION="1.2.0"
# GitHub repository to check for releases
GITHUB_REPO="AnotherStranger/conda-poetry-dev-setup"
# The ENV File to use
CONDA_ENV_YML="${1:-env.yml}"

# Function to get the latest release version from GitHub repository
get_latest_release() {
    local repository="$1"
    latest_release=$(curl -s "https://api.github.com/repos/$repository/releases/latest" | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/')
}

# Function to ask the yes/no question and return a boolean value
ask_question() {
    local question="$1"
    while true; do
        echo "$question (yes/no)" >"$(tty)"
        read -r choice

        # Check the user input
        if [ "$choice" = "yes" ]; then
            return 0 # Return 0 for true (yes)
        elif [ "$choice" = "no" ]; then
            return 1 # Return 1 for false (no)
        else
            echo "Invalid choice. Please enter 'yes' or 'no'." >"$(tty)"
        fi
    done
}

# Get the latest release version
get_latest_release "$GITHUB_REPO"

# Check if the latest release version is different from the current version
if [[ "$latest_release" != "v$CURRENT_VERSION" ]]; then
    echo "A new release is available: $latest_release!"
    ask_question "Do you want to update?"
    ask_question_result=$?
    echo "Result: $ask_question_result"

    if [ $ask_question_result -eq 0 ]; then
        wget "https://github.com/$GITHUB_REPO/releases/download/$latest_release/dev-setup.sh" -O "dev-setup-next.sh"
        chmod +x "dev-setup-next.sh"
        mv "dev-setup-next.sh" "dev-setup.sh"
        echo "Update successfully. Please re-run the script."
    else
        echo "Update skipped."
    fi
else
    echo "No new releases found."
fi

# Ensure script is called correctly using source
# Ignore shellcheck warining. not accessing an index is intentional here
# shellcheck disable=SC2128
if [ "$0" != "$BASH_SOURCE" ]; then
    echo "Performing Development setup..."
else
    echo "You need to call this script using the command source. Please execute:"
    echo "source $0"
    exit 1
fi

# Find the conda command to use (conda or mamba)
if type "mamba" >/dev/null 2>&1; then
    CONDA_CMD="mamba"
elif type "conda" >/dev/null 2>&1; then
    CONDA_CMD="conda"
else
    echo "No conda command found"
    exit 1
fi

# Extract the conda environment name from "$CONDA_ENV_YML"
CONDA_ENV=$(grep 'name:' "$CONDA_ENV_YML" | cut -f 2 -d ' ')

# Check if the conda environment already exists
if $CONDA_CMD env list | grep -q "$CONDA_ENV"; then
    $CONDA_CMD env update -f "$CONDA_ENV_YML"
else
    $CONDA_CMD env create -f "$CONDA_ENV_YML"
fi

# Activate the conda environment and execute poetry install
if $CONDA_CMD activate "$CONDA_ENV" 2>/dev/null || conda activate "$CONDA_ENV" 2>/dev/null; then
    poetry config --local virtualenvs.create false
    poetry update
    poetry install
else
    echo "Failed to activate the conda environment."
fi

echo "Environment setup finished successfully."
echo "Please ensure to set python interpreter of your IDE to the conda environment $CONDA_ENV"
}; { main "$@" ;}
