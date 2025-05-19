git diff --name-only --diff-filter=U | while read file; do
  awk '/^<<<<<<< HEAD$/ {skip=1; next}
       /^=======/ {skip=0; keep=1; next}
       /^>>>>>>>/ {keep=0; next}
       skip == 0 && keep == 1' "$file" > tmp && mv tmp "$file"
done
