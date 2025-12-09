# PowerShell 自动提交脚本
# 使用方法: .\auto_commit.ps1 "提交信息"

param(
    [Parameter(Mandatory=$false)]
    [string]$Message = "chore: 自动提交更改"
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
Write-Host "将要提交以下文件:" -ForegroundColor Cyan
git status --short

# 添加所有更改
Write-Host "`n添加所有更改..." -ForegroundColor Cyan
git add -A

# 提交更改
Write-Host "提交更改: $Message" -ForegroundColor Cyan
git commit -m $Message

# 获取当前分支
$branch = git rev-parse --abbrev-ref HEAD

# 询问是否推送到远程
$push = Read-Host "是否推送到远程仓库 origin/$branch? (y/n)"
if ($push -eq 'y' -or $push -eq 'Y') {
    Write-Host "推送到远程仓库..." -ForegroundColor Cyan
    git push origin $branch
} else {
    Write-Host "跳过推送" -ForegroundColor Yellow
}

Write-Host "`n完成!" -ForegroundColor Green

