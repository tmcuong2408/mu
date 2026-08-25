@echo off
chcp 65001 > nul
cd /d "%~dp0"

:: Lấy ngày giờ hệ thống dạng YYYY-MM-DD HH:MM
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set YYYY=%datetime:~0,4%
set MM=%datetime:~4,2%
set DD=%datetime:~6,2%
set HH=%datetime:~8,2%
set Min=%datetime:~10,2%

set CURRENT_DATE=%YYYY%-%MM%-%DD% %HH%:%Min%

echo ========================================
echo   Auto Commit & Push: %CURRENT_DATE%
echo ========================================

:: Thêm tất cả thay đổi
git add .

:: Commit với ngày tháng
git commit -m "Auto backup: %CURRENT_DATE%"

:: Push lên branch hiện tại
git push

echo ========================================
echo   Hoan thanh!
echo ========================================
pause