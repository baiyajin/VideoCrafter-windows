# PowerShell 自动提交所有更改脚本
# 使用方法: .\auto_commit_all.ps1 "提交信息"
# 如果不提供提交信息，将使用默认信息

param(
    [Parameter(Mandatory=$false)]
    [string]$Message = ""
)

# 切换到项目根目录
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

# 检查是否有未提交的更改
$status = git status --porcelain
if ([string]::IsNullOrWhiteSpace($status)) {
    Write-Host "没有未提交的更改" -ForegroundColor Yellow
    exit 0
}

# 显示将要提交的文件
Write-Host "`n检测到以下更改:" -ForegroundColor Cyan
git status --short

# 如果没有提供提交信息，生成默认信息
if ([string]::IsNullOrWhiteSpace($Message)) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $changedFiles = (git status --short | Measure-Object -Line).Lines
    $Message = "chore: 自动提交更改 ($changedFiles 个文件) - $timestamp"
}

# 添加所有更改
Write-Host "`n添加所有更改..." -ForegroundColor Cyan
git add -A

# 提交更改
Write-Host "提交更改: $Message" -ForegroundColor Cyan
git commit -m $Message

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n提交成功!" -ForegroundColor Green
    
    # 获取当前分支
    $branch = git rev-parse --abbrev-ref HEAD
    
    # 询问是否推送到远程
    $push = Read-Host "`n是否推送到远程仓库 origin/$branch? (y/n)"
    if ($push -eq 'y' -or $push -eq 'Y') {
        Write-Host "推送到远程仓库..." -ForegroundColor Cyan
        git push origin $branch
        if ($LASTEXITCODE -eq 0) {
            Write-Host "推送成功!" -ForegroundColor Green
        } else {
            Write-Host "推送失败，请检查网络连接或远程仓库配置" -ForegroundColor Red
        }
    } else {
        Write-Host "跳过推送" -ForegroundColor Yellow
    }
} else {
    Write-Host "`n提交失败，请检查错误信息" -ForegroundColor Red
    exit 1
}

