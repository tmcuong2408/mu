:: Add all changes
git add .

:: Commit with date and time
git commit -m "Auto backup: %DATE% %TIME%"

:: Push to current branch
git push

echo ========================================
echo   Completed!
echo ========================================
pause