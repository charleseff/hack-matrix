# Dev container: Swift + Python for HackMatrix
FROM swift:6.0.3-jammy

# Install Python and dev tools
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    git \
    sudo \
    iptables \
    ipset \
    aggregate \
    dnsutils \
    jq \
    gettext-base \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies are installed in workspace venv (python/venv-linux)
# See devcontainer.json for venv setup in postCreateCommand

# Create non-root user (devcontainer convention)
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Create directory for persistent zsh history (mounted as volume in devcontainer.json)
RUN mkdir -p /commandhistory && chown -R $USERNAME /commandhistory

# Set up firewall script with passwordless sudo
COPY .devcontainer/init-firewall.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/init-firewall.sh && \
    echo "$USERNAME ALL=(root) NOPASSWD: /usr/local/bin/init-firewall.sh" > /etc/sudoers.d/$USERNAME-firewall && \
    chmod 0440 /etc/sudoers.d/$USERNAME-firewall

USER $USERNAME
