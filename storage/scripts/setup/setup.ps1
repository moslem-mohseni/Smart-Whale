# setup.ps1
# Script for setting up Smart Whale system with sequential service startup

# Colors for output
$Green = "Green"
$Cyan = "Cyan"
$Yellow = "Yellow"
$Red = "Red"

# Get project root directory
$ProjectRoot = Get-Location

# Test prerequisites
function Test-Prerequisites {
    Write-Host "Checking prerequisites..." -ForegroundColor $Cyan

    # Check Docker
    if (-not (Get-Command "docker" -ErrorAction SilentlyContinue)) {
        Write-Host "Docker is not installed. Please install Docker Desktop." -ForegroundColor $Red
        return $false
    }

    # Check Docker Compose
    if (-not (Get-Command "docker-compose" -ErrorAction SilentlyContinue)) {
        Write-Host "Docker Compose is not installed. Please install Docker Compose." -ForegroundColor $Red
        return $false
    }

    Write-Host "All prerequisites are met." -ForegroundColor $Green
    return $true
}

# Create required directories
function New-Directories {
    Write-Host "Creating required directories..." -ForegroundColor $Cyan

    # Create storage directories
    $dirs = @(
        "storage/logs/app",
        "storage/shared/tmp",
        "storage/shared/uploads",
        "storage/shared/db/clickhouse",
        "storage/shared/db/timescaledb",
        "storage/shared/db/milvus",
        "storage/shared/kafka/data",
        "storage/shared/kafka/zookeeper",
        "storage/shared/cache/redis",
        "storage/config/prometheus",
        "storage/config/clickhouse",
        "storage/config/grafana/provisioning"
    )

    foreach ($dir in $dirs) {
        $path = Join-Path -Path $ProjectRoot -ChildPath $dir

        if (-not (Test-Path $path)) {
            New-Item -ItemType Directory -Force -Path $path | Out-Null
            Write-Host "  Created: $dir" -ForegroundColor $Green
        }
    }

    Write-Host "Directories created successfully." -ForegroundColor $Green
}

# Stop existing services
function Stop-Services {
    Write-Host "Stopping existing services..." -ForegroundColor $Cyan

    docker-compose down --remove-orphans

    Write-Host "Existing services stopped." -ForegroundColor $Green
}

# Start Docker services sequentially
function Start-Services {
    Write-Host "Starting Docker services..." -ForegroundColor $Cyan

    # Start infrastructure services first
    Write-Host "  Starting Zookeeper..." -ForegroundColor $Yellow
    docker-compose up -d zookeeper

    # Wait for Zookeeper to be ready
    Write-Host "  Waiting for Zookeeper to initialize (15 seconds)..." -ForegroundColor $Yellow
    Start-Sleep -Seconds 15

    # Start Kafka
    Write-Host "  Starting Kafka..." -ForegroundColor $Yellow
    docker-compose up -d kafka

    # Wait for Kafka to be ready
    Write-Host "  Waiting for Kafka to initialize (15 seconds)..." -ForegroundColor $Yellow
    Start-Sleep -Seconds 15

    # Start database services
    Write-Host "  Starting TimescaleDB and Redis..." -ForegroundColor $Yellow
    docker-compose up -d timescaledb redis

    # Wait for database services
    Write-Host "  Waiting for database services to initialize (10 seconds)..." -ForegroundColor $Yellow
    Start-Sleep -Seconds 10

    # Start remaining services
    Write-Host "  Starting remaining services..." -ForegroundColor $Yellow
    docker-compose up -d

    # Check if all services are running
    $services = docker-compose ps --services
    $allRunning = $true

    foreach ($service in $services) {
        $status = docker-compose ps --format json $service | ConvertFrom-Json | Select-Object -ExpandProperty State
        if ($status -notmatch "running") {
            Write-Host "  Service $service is not running: $status" -ForegroundColor $Red
            $allRunning = $false
        }
    }

    if (-not $allRunning) {
        Write-Host "Some services failed to start. Check docker-compose logs for details." -ForegroundColor $Red
        return $false
    }

    Write-Host "Docker services started successfully." -ForegroundColor $Green
    return $true
}

# Install dependencies
function Install-Dependencies {
    Write-Host "Installing dependencies..." -ForegroundColor $Cyan

    Write-Host "Waiting for services to initialize (10 seconds)..." -ForegroundColor $Yellow
    Start-Sleep -Seconds 10

    docker exec -i smart_whale-app-1 pip install --no-cache-dir tzlocal==4.2 confluent-kafka==2.0.2 pymilvus==2.5.5

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to install dependencies." -ForegroundColor $Red
        return $false
    }

    Write-Host "Dependencies installed successfully." -ForegroundColor $Green
    return $true
}

# Initialize database schemas
function Initialize-Schemas {
    Write-Host "Initializing database schemas..." -ForegroundColor $Cyan

    docker exec -i smart_whale-app-1 python storage/scripts/init_schema.py --all

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to initialize database schemas." -ForegroundColor $Red
        return $false
    }

    Write-Host "Database schemas initialized successfully." -ForegroundColor $Green
    return $true
}

# Display access information
function Show-AccessInfo {
    Write-Host "Access Information:" -ForegroundColor $Cyan
    Write-Host "  Application: http://localhost:8000" -ForegroundColor $Yellow
    Write-Host "  ClickHouse UI: http://localhost:8123" -ForegroundColor $Yellow
    Write-Host "  Grafana: http://localhost:6504" -ForegroundColor $Yellow
    Write-Host "  Prometheus: http://localhost:8090" -ForegroundColor $Yellow
    Write-Host "  Minio: http://localhost:9010" -ForegroundColor $Yellow
}

# Main function
function Start-Setup {
    Write-Host "Starting Smart Whale Setup" -ForegroundColor $Cyan
    Write-Host "============================"

    # Check prerequisites
    if (-not (Test-Prerequisites)) {
        return
    }

    # Create required directories
    New-Directories

    # Stop existing services
    Stop-Services

    # Clean up previous data if needed
    Write-Host "Cleaning up previous data..." -ForegroundColor $Cyan
    if (Test-Path "storage/shared/kafka/data") {
        Remove-Item -Path "storage/shared/kafka/data/*" -Recurse -Force -ErrorAction SilentlyContinue
    }
    if (Test-Path "storage/shared/kafka/zookeeper") {
        Remove-Item -Path "storage/shared/kafka/zookeeper/*" -Recurse -Force -ErrorAction SilentlyContinue
    }
    Write-Host "Cleanup completed." -ForegroundColor $Green

    # Start Docker services
    if (-not (Start-Services)) {
        return
    }

    # Install dependencies
    if (-not (Install-Dependencies)) {
        return
    }

    # Initialize database schemas
    if (-not (Initialize-Schemas)) {
        return
    }

    # Show access information
    Show-AccessInfo

    Write-Host "Smart Whale setup completed successfully!" -ForegroundColor $Green
    Write-Host "Run 'docker-compose ps' to see running services." -ForegroundColor $Cyan
}

# Run the script
Start-Setup
