# Section 1: Build/Install
# This section is for first-time setup and installations.

install_dependencies() {
    # Function to install packages on macOS
    install_mac() {
        which brew > /dev/null
        if [ $? -ne 0 ]; then
            echo "Installing Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        echo "Updating Homebrew packages..."
        brew update
        echo "Installing required packages..."
        brew install git curl make node python
        # Verify installations
        echo "Node.js version: $(node --version)"
        echo "npm version: $(npm --version)"
        # Install PM2 globally
        npm install pm2 -g
        git clone https://github.com/web-genie-ai/web-genie-ai.git
        cd web-genie-ai
        
        # Create and activate virtual environment
        python3 -m venv .venv
        source .venv/bin/activate
        
        # Install uv package manager
        curl -LsSf https://astral.sh/uv/install.sh | sh
        echo "Installing dependencies..."
        uv sync

        # Install Chrome and Playwright dependencies
        npm install -g lighthouse
        brew install --cask google-chrome
        
        # Install Playwright and its dependencies
        playwright install-deps
        playwright install
    }

    # Function to install packages on Ubuntu/Debian
    install_ubuntu() {
        echo "Updating system packages..."
        sudo apt update
        echo "Installing required packages..."
        sudo apt install --assume-yes make curl python3-pip
        # Install Node.js and npm
        curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
        sudo apt install --assume-yes nodejs
        # Verify installations
        echo "Node.js version: $(node --version)"
        echo "npm version: $(npm --version)"
        # Install PM2 globally
        npm install pm2 -g
        
        # Install uv package manager and add to PATH
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.cargo/bin:$PATH"
        echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
        echo "Installing dependencies..."
        uv sync

        # Install Chrome and Playwright dependencies
        npm install -g lighthouse
        wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
        sudo apt install --assume-yes gdebi-core
        sudo gdebi -n google-chrome-stable_current_amd64.deb
        rm google-chrome-stable_current_amd64.deb
        
        # Create and activate virtual environment
        source .venv/bin/activate

        # Install Playwright and its dependencies
        playwright install-deps
        playwright install
    }

    # Detect OS and call the appropriate function
    if [[ "$OSTYPE" == "darwin"* ]]; then
        install_mac
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        install_ubuntu
    else
        echo "Unsupported operating system."
        exit 1
    fi

    # Update your shell's source to include Cargo's path
    source "$HOME/.cargo/env"
}

# Call install_dependencies only if it's the first time running the script
if [ ! -f ".dependencies_installed" ]; then
    install_dependencies
    touch .dependencies_installed
fi