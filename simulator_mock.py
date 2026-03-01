#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import random
import time
from collections import deque
import json

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)  # 行缓冲

"""
===============================================================================
simulator_mock.py - PID 调参模拟器 (Mock LLM 版)

用于演示项目功能，无需真实 API Key。
模拟 LLM 的决策过程来进行 PID 调参。
===============================================================================
"""

# ============================================================================
# 配置
# ============================================================================

SETPOINT = 100.0          # 目标温度
INITIAL_TEMP = 20.0       # 初始温度
BUFFER_SIZE = 25           # 数据缓冲大小
MAX_ROUNDS = 20           # 最大调参轮数
CONTROL_INTERVAL = 0.05   # 控制周期 (50ms)

# PWM 输出限制
PWM_MAX = 255             # 仿真模式用 255
PWM_CHANGE_MAX = 50       # 每周期最大 PWM 变化

# PID 初始参数
kp, ki, kd = 0.5, 0.0, 0.0  # 故意设置较差的初始参数

PID_FORMULA = "standard"

# ============================================================================
# 仿真模型 (与原版保持一致)
# ============================================================================

class HeatingSimulator:
    """加热系统仿真器"""
    def __init__(self):
        self.temp = INITIAL_TEMP
        self.pwm = 0
        self.setpoint = SETPOINT
        self.integral = 0.0
        self.prev_error = 0.0
        self.timestamp = 0
        
        # 仿真参数
        self.heater_temp = INITIAL_TEMP
        self.ambient_temp = 20.0
        self.heater_coeff = 300.0      # 加热器能力
        self.heat_transfer = 0.1       # 传热系数
        self.cooling_coeff = 0.05      # 散热系数
        self.noise_level = 0.1         # 噪声
        
        self.prev_pwm = 0
        self.prev_prev_error = 0.0
        self.last_pid_output = 0.0
    
    def compute_pid(self):
        error = self.setpoint - self.temp
        self.integral += error * CONTROL_INTERVAL
        self.integral = max(-200, min(200, self.integral))
        derivative = (error - self.prev_error) / CONTROL_INTERVAL
        
        # 标准 PID
        pid_output = kp * error + ki * self.integral + kd * derivative
        
        # PWM 变化率限制
        pwm_delta = pid_output - self.prev_pwm
        if abs(pwm_delta) > PWM_CHANGE_MAX:
            pid_output = self.prev_pwm + (PWM_CHANGE_MAX if pwm_delta > 0 else -PWM_CHANGE_MAX)
        
        self.pwm = max(0, min(PWM_MAX, pid_output))
        self.prev_pwm = self.pwm
        
        self.prev_prev_error = self.prev_error
        self.prev_error = error
        
    def update(self):
        # 简单的热力学模型
        # 加热器升温
        target_heater_temp = self.ambient_temp + (self.pwm / 255.0) * self.heater_coeff
        self.heater_temp += (target_heater_temp - self.heater_temp) * 0.1
        
        # 物体升温
        heat_in = (self.heater_temp - self.temp) * self.heat_transfer
        heat_out = (self.temp - self.ambient_temp) * self.cooling_coeff
        
        self.temp += (heat_in - heat_out) * CONTROL_INTERVAL
        
        # 噪声
        self.temp += random.gauss(0, self.noise_level)
        self.timestamp += int(CONTROL_INTERVAL * 1000)

    def get_data(self):
        return {
            "timestamp": self.timestamp,
            "setpoint": self.setpoint,
            "input": self.temp,
            "pwm": self.pwm,
            "error": self.setpoint - self.temp,
            "p": kp,
            "i": ki,
            "d": kd
        }

# ============================================================================
# Mock LLM API 调用
# ============================================================================

def call_llm_mock(metrics: dict) -> dict:
    """模拟 LLM 分析"""
    print("\n[MockLLM] 正在分析数据...")
    time.sleep(1)  # 模拟网络延迟
    
    avg_error = metrics['avg_error']
    max_error = metrics['max_error']
    current_temp = metrics['latest_temp']
    
    analysis = ""
    new_p, new_i, new_d = kp, ki, kd
    
    # 简单的规则库模拟 AI 决策
    if avg_error > 10:
        if current_temp < SETPOINT:
            analysis = "温度过低，显著增加比例系数 P 以加快升温。"
            new_p += 0.5
            new_i += 0.05
        else:
            analysis = "温度过高，减小 P，增加 D 抑制超调。"
            new_p = max(0.1, new_p - 0.2)
            new_d += 0.1
    elif avg_error > 2:
        if current_temp < SETPOINT:
            analysis = "接近目标，微调增加 I 消除静差。"
            new_i += 0.02
            new_p += 0.1
        else:
            analysis = "有轻微超调，增加 D。"
            new_d += 0.05
    else:
        analysis = "误差很小，参数表现良好，保持微调。"
        status = "DONE" if avg_error < 0.5 else "TUNING"
        return {
            "analysis": analysis,
            "p": kp, "i": ki, "d": kd,
            "status": status
        }
        
    return {
        "analysis": analysis,
        "p": round(new_p, 4),
        "i": round(new_i, 4),
        "d": round(new_d, 4),
        "status": "TUNING"
    }

# ============================================================================
# 主循环
# ============================================================================

buffer = deque(maxlen=BUFFER_SIZE)
sim = HeatingSimulator()

def run_tuning():
    global kp, ki, kd
    rounds = 0
    
    print("\n" + "="*60)
    print(f"开始 PID 自动调参实验 (Mock Mode)")
    print("="*60)
    
    while rounds < MAX_ROUNDS:
        rounds += 1
        
        # 1. 采集数据
        print(f"\n[第 {rounds} 轮] 采集数据中...")
        for i in range(BUFFER_SIZE):
            sim.compute_pid()
            sim.update()
            data = sim.get_data()
            buffer.append(data)
            if i % 5 == 0:
                print(f"  t={data['timestamp']}ms T={data['input']:.1f}°C (Target:{SETPOINT}) PWM={data['pwm']:.0f}")
            time.sleep(0.01) # 加速仿真
            
        # 2. 计算指标
        errors = [abs(d['error']) for d in buffer]
        metrics = {
            'avg_error': sum(errors) / len(errors),
            'max_error': max(errors),
            'latest_temp': buffer[-1]['input']
        }
        
        print(f"  >> 平均误差: {metrics['avg_error']:.2f}°C")
        
        # 3. 判断是否完成
        if metrics['avg_error'] < 0.5:
            print("\n✅ 调参完成！")
            break
            
        # 4. 调用 Mock LLM
        result = call_llm_mock(metrics)
        
        kp = result['p']
        ki = result['i']
        kd = result['d']
        
        print(f"[MockLLM] 分析: {result['analysis']}")
        print(f"[MockLLM] 新参数: P={kp}, I={ki}, D={kd}")
        
    print("\n" + "="*60)
    print(f"最终参数: P={kp}, I={ki}, D={kd}")
    print(f"最终温度: {metrics['latest_temp']:.2f}°C")

if __name__ == "__main__":
    run_tuning()
