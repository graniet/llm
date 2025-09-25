
#strace_file=`date -u +"%Y-%m-%dT%H:%M:%SZ"`
#strace -f -o logs/strace_${strace_file}.txt -s 9999
~/gemini-cli/bundle/gemini.js --output-format json \
			      --approval-mode yolo \
			      --model gemini-2.5-flash \
			      --checkpointing \
			      --include-directories ~/nix \
			      --debug \
			      --prompt  'follow instructions in @crq.txt'
