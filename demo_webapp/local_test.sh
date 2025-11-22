#!/bin/bash
# Local testing script for GTMØ Web Demo

echo "🧪 GTMØ Web Demo - Local Test"
echo "========================================"
echo ""

# Check if ANTHROPIC_API_KEY is set
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "⚠️  Warning: ANTHROPIC_API_KEY not set"
    echo "   Recommendations will be disabled"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Start backend
echo "1️⃣ Starting backend (port 8000)..."
cd demo_webapp/api

# Install dependencies if needed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "   Installing dependencies..."
    pip install -r requirements.txt
fi

# Start server in background
python main.py &
BACKEND_PID=$!

# Wait for backend to start
echo "   Waiting for backend to start..."
sleep 3

# Check if backend is running
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo "   ❌ Backend failed to start"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

echo "   ✓ Backend running (PID: $BACKEND_PID)"
echo ""

# Start frontend
cd ../docs
echo "2️⃣ Starting frontend (port 8080)..."
python -m http.server 8080 &
FRONTEND_PID=$!

sleep 2

echo "   ✓ Frontend running (PID: $FRONTEND_PID)"
echo ""

# Display info
echo "========================================"
echo "✅ GTMØ Web Demo is running!"
echo ""
echo "📍 URLs:"
echo "   Backend:  http://localhost:8000"
echo "   Frontend: http://localhost:8080"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "🧪 Test file: sample_document.txt"
echo ""
echo "Press Ctrl+C to stop servers"
echo "========================================"

# Cleanup function
cleanup() {
    echo ""
    echo "🛑 Stopping servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "   ✓ Stopped"
    exit 0
}

trap cleanup INT TERM

# Wait for user to stop
wait
