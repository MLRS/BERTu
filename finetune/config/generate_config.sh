#!/bin/bash

set -euo pipefail

for model in bertu mbert mbertu; do
    for task in dp upos xpos ner sa; do
        config_name="autogen/${task}_${model}.jsonnet"
        echo $config_name
        echo "local task_builder = import \"../lib/tasks.libsonnet\";" > $config_name
        echo "" >> $config_name

        args=""
        if [ ${task} == "upos" ];
        then
            args=" , language_specific=false"
            task="pos"
        elif [ ${task} == "xpos" ];
        then
            args=" , language_specific=true"
            task="pos"
        fi
        echo "task_builder.build_${task}(\"${model}\"${args})" >> $config_name
    done
done
