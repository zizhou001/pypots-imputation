#!/bin/bash



cd /root/autodl-tmp/project/

# 遍历 missing-rate 和 max-missing-rate 的所有组合
for MISSING_RATE in 0.2 0.4 0.6 0.8; do
    for MAX_MISSING_RATE in 0.3 0.4 0.5 0.6; do
        echo "Running with missing-rate=$MISSING_RATE and max-missing-rate=$MAX_MISSING_RATE"

        # 计算 SEQ_LENGTH
        PRODUCT=$(echo "scale=2; $MISSING_RATE * $MAX_MISSING_RATE * 1000" | bc)

        if (( $(echo "$PRODUCT <= 64" | awk '{print ($1 <= 64)}') )); then
            SEQ_LENGTH=64
        elif (( $(echo "$PRODUCT <= 128" | awk '{print ($1 <= 128)}') )); then
            SEQ_LENGTH=128
        elif (( $(echo "$PRODUCT <= 256" | awk '{print ($1 <= 256)}') )); then
            SEQ_LENGTH=256
        else
            SEQ_LENGTH=256
        fi

        echo "Using SEQ_LENGTH=$SEQ_LENGTH"

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
