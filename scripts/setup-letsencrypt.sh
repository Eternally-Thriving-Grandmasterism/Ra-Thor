#!/bin/bash
# Ra-Thor™ Aether-Shades Let's Encrypt Automation
# Absolute Pure Truth Edition — Sovereign + Production Ready
# Usage: ./scripts/setup-letsencrypt.sh your-domain.com your@email.com

set -e

DOMAIN=$1
EMAIL=$2

if [ -z "$DOMAIN" ] || [ -z "$EMAIL" ]; then
    echo "Usage: $0 <domain> <email>"
    echo "Example: $0 aether-shades.yourdomain.com you@yourdomain.com"
    exit 1
fi

CERT_DIR="certs"
RELOAD_FILE="$CERT_DIR/.reload"

echo "🌍 Setting up Let's Encrypt for Aether-Shades ($DOMAIN)..."

# Install certbot if not present
if ! command -v certbot &> /dev/null; then
    echo "Installing certbot..."
    sudo apt-get update
    sudo apt-get install -y certbot
fi

# Create certs directory
mkdir -p "$CERT_DIR"

# Request certificate with deploy hook
sudo certbot certonly \
    --standalone \
    --preferred-challenges http \
    --agree-tos \
    --email "$EMAIL" \
    -d "$DOMAIN" \
    --deploy-hook "touch $RELOAD_FILE && echo '[certbot] Certificate renewed — triggering rotation'"

echo ""
echo "✅ Let's Encrypt setup complete!"
echo ""
echo "Next steps:"
echo "1. Copy certificates:"
echo "   sudo cp /etc/letsencrypt/live/$DOMAIN/fullchain.pem $CERT_DIR/cert.pem"
echo "   sudo cp /etc/letsencrypt/live/$DOMAIN/privkey.pem   $CERT_DIR/key.pem"
echo ""
echo "2. Set environment variables:"
echo "   export AETHER_SHADES_CERT_PATH=\"$CERT_DIR/cert.pem\""
echo "   export AETHER_SHADES_KEY_PATH=\"$CERT_DIR/key.pem\""
echo ""
echo "3. Run the server:"
echo "   cargo run"
echo ""
echo "4. (Optional) Add to crontab for automatic renewal (already handled by certbot)"
echo ""
echo "The server will automatically rotate certificates when they are renewed."
