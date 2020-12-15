
# if running on bash instead of zsh console, rename this file to "create_new_experiment.sh"

# help function
function help_ {
   cat << EOF
usage: create_new_experiment.zsh -n/--name <epxeriment-name>
EOF
   exit 1
}

# check if there are two flags
if [ $# -ne 2 ]; then
    help_;
fi
# check if the first flag is either -n or --name
if [ "$1" != "-n" ] && [ "$1" != "--name" ]; then
    help_;
fi

# finally create new experiment environment
cp -r experiments/template_experiment experiments/$2