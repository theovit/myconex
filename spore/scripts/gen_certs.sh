#!/bin/bash
# MYCONEX Certificate Generation Script
# Generates mTLS certificates for secure mesh communication

set -e

# Configuration
CERT_DIR="/home/techno-shaman/myconex/spore/certs"
CA_KEY="${CERT_DIR}/ca.key"
CA_CERT="${CERT_DIR}/ca.crt"
SERVER_KEY="${CERT_DIR}/server.key"
SERVER_CERT="${CERT_DIR}/server.crt"
CLIENT_KEY="${CERT_DIR}/client.key"
CLIENT_CERT="${CERT_DIR}/client.crt"
CONFIG_FILE="${CERT_DIR}/openssl.cnf"

# Certificate details
COUNTRY="US"
STATE="California"
CITY="San Francisco"
ORG="MYCONEX"
OU="AI Mesh"
EMAIL="admin@myconex.local"
DAYS=3650

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Create certificate directory
create_cert_dir() {
    log "Creating certificate directory: $CERT_DIR"
    mkdir -p "$CERT_DIR"
    chmod 700 "$CERT_DIR"
}

# Generate OpenSSL configuration
generate_openssl_config() {
    log "Generating OpenSSL configuration"

    cat > "$CONFIG_FILE" << EOF
[ req ]
distinguished_name = req_distinguished_name
req_extensions = v3_req
prompt = no

[ req_distinguished_name ]
C = $COUNTRY
ST = $STATE
L = $CITY
O = $ORG
OU = $OU
CN = myconex.local
emailAddress = $EMAIL

[ v3_req ]
keyUsage = keyEncipherment, dataEncipherment
extendedKeyUsage = serverAuth, clientAuth
subjectAltName = @alt_names

[ alt_names ]
DNS.1 = localhost
DNS.2 = myconex.local
DNS.3 = *.myconex.local
IP.1 = 127.0.0.1
IP.2 = ::1

[ v3_ca ]
subjectKeyIdentifier = hash
authorityKeyIdentifier = keyid:always,issuer
basicConstraints = critical,CA:true
keyUsage = critical,digitalSignature,keyCertSign,cRLSign

[ v3_server ]
subjectKeyIdentifier = hash
authorityKeyIdentifier = keyid,issuer
basicConstraints = CA:FALSE
keyUsage = digitalSignature,keyEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[ v3_client ]
subjectKeyIdentifier = hash
authorityKeyIdentifier = keyid,issuer
basicConstraints = CA:FALSE
keyUsage = digitalSignature,keyEncipherment
extendedKeyUsage = clientAuth
EOF
}

# Generate CA certificate
generate_ca() {
    log "Generating Certificate Authority (CA)"

    # Generate CA private key
    openssl genrsa -out "$CA_KEY" 4096
    chmod 600 "$CA_KEY"

    # Generate CA certificate
    openssl req -new -x509 -days $DAYS -key "$CA_KEY" -sha256 -out "$CA_CERT" \
        -subj "/C=$COUNTRY/ST=$STATE/L=$CITY/O=$ORG/OU=$OU/CN=MYCONEX CA/emailAddress=$EMAIL" \
        -extensions v3_ca -config "$CONFIG_FILE"

    chmod 644 "$CA_CERT"
    log "CA certificate generated: $CA_CERT"
}

# Generate server certificate
generate_server_cert() {
    log "Generating server certificate"

    # Generate server private key
    openssl genrsa -out "$SERVER_KEY" 2048
    chmod 600 "$SERVER_KEY"

    # Generate certificate signing request
    openssl req -new -key "$SERVER_KEY" -out "${CERT_DIR}/server.csr" \
        -subj "/C=$COUNTRY/ST=$STATE/L=$CITY/O=$ORG/OU=$OU/CN=myconex.local/emailAddress=$EMAIL" \
        -config "$CONFIG_FILE"

    # Sign server certificate
    openssl x509 -req -days $DAYS -in "${CERT_DIR}/server.csr" \
        -CA "$CA_CERT" -CAkey "$CA_KEY" -CAcreateserial \
        -out "$SERVER_CERT" -sha256 -extensions v3_server -extfile "$CONFIG_FILE"

    chmod 644 "$SERVER_CERT"
    rm "${CERT_DIR}/server.csr"
    log "Server certificate generated: $SERVER_CERT"
}

# Generate client certificate
generate_client_cert() {
    log "Generating client certificate"

    # Generate client private key
    openssl genrsa -out "$CLIENT_KEY" 2048
    chmod 600 "$CLIENT_KEY"

    # Generate certificate signing request
    openssl req -new -key "$CLIENT_KEY" -out "${CERT_DIR}/client.csr" \
        -subj "/C=$COUNTRY/ST=$STATE/L=$CITY/O=$ORG/OU=$OU/CN=myconex-client/emailAddress=$EMAIL" \
        -config "$CONFIG_FILE"

    # Sign client certificate
    openssl x509 -req -days $DAYS -in "${CERT_DIR}/client.csr" \
        -CA "$CA_CERT" -CAkey "$CA_KEY" -CAcreateserial \
        -out "$CLIENT_CERT" -sha256 -extensions v3_client -extfile "$CONFIG_FILE"

    chmod 644 "$CLIENT_CERT"
    rm "${CERT_DIR}/client.csr"
    log "Client certificate generated: $CLIENT_CERT"
}

# Generate DH parameters for forward secrecy
generate_dh_params() {
    log "Generating DH parameters"
    openssl dhparam -out "${CERT_DIR}/dhparam.pem" 2048
    chmod 644 "${CERT_DIR}/dhparam.pem"
}

# Verify certificates
verify_certificates() {
    log "Verifying certificates"

    # Verify CA certificate
    openssl x509 -in "$CA_CERT" -text -noout > /dev/null
    log "CA certificate verified"

    # Verify server certificate
    openssl verify -CAfile "$CA_CERT" "$SERVER_CERT" > /dev/null
    log "Server certificate verified"

    # Verify client certificate
    openssl verify -CAfile "$CA_CERT" "$CLIENT_CERT" > /dev/null
    log "Client certificate verified"
}

# Generate certificate summary
generate_summary() {
    log "Generating certificate summary"

    cat > "${CERT_DIR}/README.md" << EOF
# MYCONEX mTLS Certificates

This directory contains the certificates for mutual TLS authentication in the MYCONEX mesh.

## Files

- \`ca.crt\` - Certificate Authority certificate
- \`ca.key\` - Certificate Authority private key (keep secure!)
- \`server.crt\` - Server certificate
- \`server.key\` - Server private key
- \`client.crt\` - Client certificate
- \`client.key\` - Client private key
- \`dhparam.pem\` - DH parameters for forward secrecy
- \`openssl.cnf\` - OpenSSL configuration used for generation

## Certificate Details

**Certificate Authority:**
$(openssl x509 -in "$CA_CERT" -text -noout | grep "Subject:" | sed 's/Subject: //')

**Server Certificate:**
$(openssl x509 -in "$SERVER_CERT" -text -noout | grep "Subject:" | sed 's/Subject: //')
Valid until: $(openssl x509 -in "$SERVER_CERT" -text -noout | grep "Not After" | sed 's/Not After : //')

**Client Certificate:**
$(openssl x509 -in "$CLIENT_CERT" -text -noout | grep "Subject:" | sed 's/Subject: //')
Valid until: $(openssl x509 -in "$CLIENT_CERT" -text -noout | grep "Not After" | sed 's/Not After : //')

## Usage

### NATS Server
\`\`\`
nats-server --tls --tlscert=server.crt --tlskey=server.key --tlsca=ca.crt
\`\`\`

### Client Connection
\`\`\`
nats --server=tls://localhost:4222 --tlscert=client.crt --tlskey=client.key --tlsca=ca.crt
\`\`\`

## Security Notes

- Keep private keys (\`.key\` files) secure and never share them
- The CA private key should only be used for signing new certificates
- Rotate certificates regularly (at least annually)
- Use strong passphrases for private keys in production

Generated on: $(date)
EOF

    log "Certificate summary generated: ${CERT_DIR}/README.md"
}

# Main execution
main() {
    log "Starting MYCONEX certificate generation"

    # Check if certificates already exist
    if [[ -f "$CA_CERT" ]]; then
        warn "Certificates already exist. Backing up existing certificates."
        backup_dir="${CERT_DIR}/backup-$(date +%Y%m%d-%H%M%S)"
        mkdir -p "$backup_dir"
        cp "$CERT_DIR"/* "$backup_dir"/ 2>/dev/null || true
        rm -f "$CERT_DIR"/*
    fi

    create_cert_dir
    generate_openssl_config
    generate_ca
    generate_server_cert
    generate_client_cert
    generate_dh_params
    verify_certificates
    generate_summary

    log "Certificate generation completed successfully!"
    log "Certificates are located in: $CERT_DIR"
    log "Review the README.md file for usage instructions."
}

# Run main function
main "$@"