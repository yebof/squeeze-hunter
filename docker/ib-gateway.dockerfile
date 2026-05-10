FROM gnzsnz/ib-gateway:stable

# Default to paper, port 7497.
ENV TRADING_MODE=paper \
    TWOFA_TIMEOUT_ACTION=restart \
    AUTO_RESTART_TIME=00:00 \
    READ_ONLY_API=no \
    BYPASS_WARNING=yes

EXPOSE 7497 5900
