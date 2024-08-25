#!/bin/bash

SEQ_LENGTH=64

cd /root/autodl-tmp/project/

# 遍历 missing-rate 和 max-missing-rate 的所有组合
for MISSING_RATE in 0.2 0.4 0.6 0.8; do
    for MAX_MISSING_RATE in 0.1 0.15 0.2 0.25; do
#        echo "Running with missing-rate=$MISSING_RATE and max-missing-rate=$MAX_MISSING_RATE"

#        echo "Using SEQ_LENGTH=$SEQ_LENGTH"

        # 执行 Python 程序
        python /root/autodl-tmp/project/main.py \
            --seq-length $SEQ_LENGTH \
            --missing-rate $MISSING_RATE \
            --max-missing-rate $MAX_MISSING_RATE

        # 检查 Python 脚本的退出状态
        if [ $? -ne 0 ]; then
            echo "Error occurred while running the script with missing-rate=$MISSING_RATE and max-missing-rate=$MAX_MISSING_RATE"
            exit 1
        fi
    done
done

# 关机
shutdown -h now
