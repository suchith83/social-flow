# Social Flow Backend - Database Setup Script
# Sets up PostgreSQL using Docker

Write-Host "================================" -ForegroundColor Cyan
Write-Host "Social Flow - Database Setup" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Check Docker
Write-Host "Checking Docker..." -ForegroundColor Yellow
try {
    docker --version | Out-Null
    Write-Host "[OK] Docker found" -ForegroundColor Green
}
catch {
    Write-Host "[ERROR] Docker not found" -ForegroundColor Red
    Write-Host "Install from: https://docker.com" -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# Check existing container
$exists = docker ps -a -q --filter name=social-flow-postgres
$running = docker ps -q --filter name=social-flow-postgres

if ($exists) {
    if ($running) {
        Write-Host "[OK] PostgreSQL already running" -ForegroundColor Green
    }
    else {
        Write-Host "Starting PostgreSQL..." -ForegroundColor Yellow
        docker start social-flow-postgres
        Write-Host "[OK] PostgreSQL started" -ForegroundColor Green
    }
}
else {
    Write-Host "Creating PostgreSQL container..." -ForegroundColor Yellow
    docker run --name social-flow-postgres -e POSTGRES_PASSWORD=password -e POSTGRES_DB=social_flow -p 5432:5432 -d postgres:15
    Start-Sleep 5
    Write-Host "[OK] PostgreSQL created" -ForegroundColor Green
}

Write-Host ""
Write-Host "================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Connection:" -ForegroundColor Cyan
Write-Host "  localhost:5432/social_flow" -ForegroundColor White
Write-Host "  User: postgres | Pass: password" -ForegroundColor White
Write-Host ""
Write-Host "Next: uvicorn app.main:app --reload" -ForegroundColor Cyan
Write-Host ""
