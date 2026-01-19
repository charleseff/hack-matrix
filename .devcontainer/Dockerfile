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
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (cached layer)
COPY python/requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Create non-root user (devcontainer convention)
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Set up persistent zsh history
RUN mkdir /commandhistory \
    && touch /commandhistory/.zsh_history \
    && chown -R $USERNAME /commandhistory \
    && SNIPPET="export HISTFILE=/commandhistory/.zsh_history" \
    && echo "$SNIPPET" >> "/home/$USERNAME/.zshrc"

# Set up firewall script with passwordless sudo
COPY .devcontainer/init-firewall.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/init-firewall.sh && \
    echo "$USERNAME ALL=(root) NOPASSWD: /usr/local/bin/init-firewall.sh" > /etc/sudoers.d/$USERNAME-firewall && \
    chmod 0440 /etc/sudoers.d/$USERNAME-firewall

USER $USERNAME
