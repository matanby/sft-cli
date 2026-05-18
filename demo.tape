# VHS Demo Script for sft-cli
# Run with: vhs demo.tape

Output demo.gif

# Terminal settings
Set FontSize 16
Set Width 1400
Set Height 700
Set Theme "Catppuccin Mocha"
Set Padding 20
Set Framerate 30
Set TypingSpeed 50ms
Set Shell "bash"

# Start with a clean prompt (hide the setup)
Hide
Type "export PS1='$ ' && clear"
Enter
Sleep 300ms
Show

# Start the app
Type "sft model.safetensors"
Sleep 500ms
Enter
Sleep 2s

# Navigate the tree - expand some nodes
Down
Sleep 400ms
Down
Sleep 400ms
Right
Sleep 600ms
Down
Sleep 400ms
Right
Sleep 800ms

# Go deeper into the tree
Down
Sleep 400ms
Down
Sleep 400ms
Down
Sleep 600ms

# Switch to the table panel
Tab
Sleep 1s

# Scroll through some tensors
Down
Sleep 300ms
Down
Sleep 300ms
Down
Sleep 300ms
Down
Sleep 600ms

# Show tensor details with Space
Type " "
Sleep 2s
Escape
Sleep 800ms

# Demonstrate search
Type "/"
Sleep 500ms
Type "attention"
Sleep 1.5s

# Clear search and show results
Escape
Sleep 800ms

# Demonstrate sorting
Type "s"
Sleep 800ms
Type "s"
Sleep 800ms
Type "s"
Sleep 1s

# Switch back to tree
Tab
Sleep 600ms

# Go to top
Type "g"
Sleep 600ms

# Show metadata
Type "m"
Sleep 2s
Escape
Sleep 800ms

# Quit gracefully
Type "q"
Sleep 500ms
