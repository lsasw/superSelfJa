---
title: 具身智能
icon: hand-paper
order: 98
---

# 98. 具身智能

## 概述

具身智能（Embodied AI）是指拥有物理身体（或模拟身体）并通过与环境的持续交互来实现智能行为的 AI 系统。如果说 AI for Science 让 AI 在数字世界中推动了科学发现，那么具身智能则让 AI 走出数字世界，进入了物理世界。具身智能的核心观点是：智能不仅仅是大脑中的计算，更是身体与环境互动的产物。这一理念推动了机器人学、自动驾驶和人机交互的重大进展。

### 具身智能的核心要素

| 要素 | 说明 | 传统 AI | 具身智能 |
|------|------|---------|---------|
| 身体（Body） | 感知和执行的物理载体 | 无物理身体 | 机器人、车辆、无人机 |
| 环境（Environment） | 交互的物理世界 | 静态数据集 | 动态、开放、物理约束 |
| 感知（Perception） | 传感器数据采集和处理 | 图像/文本输入 | 多传感器融合 |
| 行动（Action） | 对环境的物理操作 | 生成文本/图像 | 运动控制、操作物体 |
| 学习（Learning） | 从交互经验中改进 | 离线训练 | 在线学习、Sim-to-Real |

### 具身智能的主要应用领域

| 应用领域 | 场景描述 | 代表系统 | 技术挑战 |
|---------|---------|---------|---------|
| 家庭服务机器人 | 家务、陪伴、照料 | Figure 01、Tesla Bot | 复杂环境导航、安全操作 |
| 工业制造 | 装配、质检、物流 | 协作机器人 AMR | 高精度操作、多机器人协作 |
| 自动驾驶 | 无人驾驶汽车 | Waymo、Tesla FSD | 长尾场景、安全冗余 |
| 医疗手术 | 微创手术辅助 | 达芬奇手术机器人 | 精确控制、安全性 |
| 仓储物流 | 拣选、搬运、分拣 | Amazon Robotics | 大规模调度、效率优化 |
| 农业机器人 | 采摘、喷洒、监测 | 农业自动化的未来 | 非结构化环境适应 |
| 空间探索 | 星球探测、空间站操作 | 火星车、Canadarm | 远程操作、自主决策 |

## 具身智能技术栈

### 1. 机器人感知与理解

```python
"""
具身智能：机器人多模态感知系统
融合视觉、深度、触觉等多传感器数据
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class SensorData:
    """传感器数据"""
    rgb_image: np.ndarray          # RGB 图像 [H, W, 3]
    depth_image: np.ndarray        # 深度图像 [H, W]
    point_cloud: np.ndarray        # 点云 [N, 3]
    tactile: Optional[np.ndarray]  # 触觉传感器数据
    proprioception: np.ndarray     # 本体感知（关节角度等）

class RobotPerceptionSystem(nn.Module):
    """机器人多模态感知系统"""

    def __init__(self, embed_dim: int = 512):
        super().__init__()

        # RGB 视觉编码器
        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, embed_dim)
        )

        # 深度编码器
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, embed_dim // 2)
        )

        # 点云编码器（PointNet 风格）
        self.pointnet = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )

        # 触觉编码器
        self.tactile_encoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim // 4)
        )

        # 多模态融合
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 3 + embed_dim // 2 + embed_dim // 4, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # 物体检测头
        self.detection_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 100)  # 100 个物体类别
        )

        # 空间关系理解头
        self.spatial_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 6)  # 6 自由度位姿
        )

    def forward(self, sensor_data: SensorData) -> Dict[str, torch.Tensor]:
        """
        处理多模态传感器数据

        Args:
            sensor_data: 传感器数据
        Returns:
            感知结果
        """
        # 编码各模态
        rgb_features = self.rgb_encoder(
            torch.tensor(sensor_data.rgb_image).permute(2, 0, 1).unsqueeze(0).float()
        )

        depth_features = self.depth_encoder(
            torch.tensor(sensor_data.depth_image).unsqueeze(0).unsqueeze(0).float()
        )

        point_features = self.pointnet(
            torch.tensor(sensor_data.point_cloud).float()
        ).max(dim=0)[0].unsqueeze(0)  # 最大池化

        # 融合所有特征
        all_features = torch.cat([
            rgb_features,
            depth_features,
            point_features,
        ], dim=-1)

        if sensor_data.tactile is not None:
            tactile_features = self.tactile_encoder(
                torch.tensor(sensor_data.tactile).unsqueeze(0).float()
            )
            all_features = torch.cat([all_features, tactile_features], dim=-1)

        fused = self.fusion(all_features)

        return {
            "object_classes": self.detection_head(fused),
            "spatial_pose": self.spatial_head(fused),
            "fused_representation": fused
        }
```

### 2. 机器人运动规划与控制

```python
"""
机器人运动规划系统
包含路径规划和运动控制
"""
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class JointState:
    """关节状态"""
    positions: np.ndarray    # 关节角度 [n_joints]
    velocities: np.ndarray   # 关节速度 [n_joints]
    torques: np.ndarray      # 关节力矩 [n_joints]

@dataclass
class TrajectoryPoint:
    """轨迹点"""
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    time: float

class InverseKinematics:
    """逆运动学求解器"""

    def __init__(self, n_joints: int = 7):
        self.n_joints = n_joints
        self.damping = 0.5  # 阻尼因子

    def solve(self,
              current_joint: JointState,
              target_pose: np.ndarray,
              target_orientation: np.ndarray,
              max_iterations: int = 100,
              tolerance: float = 1e-3) -> Optional[np.ndarray]:
        """
        使用阻尼最小二乘法求解逆运动学

        Args:
            current_joint: 当前关节状态
            target_pose: 目标位置 [3]
            target_orientation: 目标朝向 [4] (四元数)
            max_iterations: 最大迭代次数
            tolerance: 收敛阈值
        Returns:
            关节角度解，或 None（无解）
        """
        joint_angles = current_joint.positions.copy()

        for i in range(max_iterations):
            # 正向运动学
            current_pose, current_orientation = self._forward_kinematics(joint_angles)

            # 计算误差
            pos_error = target_pose - current_pose
            ori_error = self._orientation_error(current_orientation, target_orientation)
            error = np.concatenate([pos_error, ori_error])

            # 检查收敛
            if np.linalg.norm(error) < tolerance:
                return joint_angles

            # 计算雅可比矩阵
            J = self._compute_jacobian(joint_angles)

            # 阻尼最小二乘解
            JTJ = J.T @ J
            damping_matrix = self.damping * np.eye(JTJ.shape[0])
            delta_theta = np.linalg.solve(JTJ + damping_matrix, J.T @ error)

            # 更新关节角度
            joint_angles += delta_theta

            # 限制关节角度范围
            joint_angles = np.clip(joint_angles, -np.pi, np.pi)

        return None  # 未收敛

    def _forward_kinematics(self, joint_angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """正向运动学：从关节角度计算末端位姿"""
        # 简化实现：假设 7 自由度机械臂
        # 实际项目应使用 DH 参数或 URDF 模型
        x = np.sum(np.cos(joint_angles)) * 0.5
        y = np.sum(np.sin(joint_angles)) * 0.5
        z = np.sum(joint_angles) * 0.1 + 0.5
        return np.array([x, y, z]), np.array([0, 0, 0, 1])

    def _compute_jacobian(self, joint_angles: np.ndarray) -> np.ndarray:
        """计算雅可比矩阵"""
        # 简化实现
        n_dof = 6  # 位置 (3) + 朝向 (3)
        J = np.zeros((n_dof, self.n_joints))
        for i in range(self.n_joints):
            J[:3, i] = np.array([-np.sin(joint_angles[i]), np.cos(joint_angles[i]), 0.1])
            J[3:, i] = np.array([0, 0, 1])
        return J

    def _orientation_error(self, current: np.ndarray, target: np.ndarray) -> np.ndarray:
        """计算朝向误差（使用四元数差）"""
        # 简化实现
        return (target[:3] - current[:3]) * 0.1


class TrajectoryPlanner:
    """轨迹规划器"""

    def generate_trajectory(self,
                            start_state: JointState,
                            goal_position: np.ndarray,
                            duration: float = 2.0,
                            n_points: int = 100) -> List[TrajectoryPoint]:
        """
        生成平滑轨迹（使用五次多项式插值）

        Args:
            start_state: 起始状态
            goal_position: 目标位置
            duration: 轨迹持续时间（秒）
            n_points: 轨迹点数
        Returns:
            轨迹点列表
        """
        trajectory = []
        dt = duration / n_points

        for i in range(n_points):
            t = i * dt
            s = t / duration  # 归一化时间 [0, 1]

            # 五次多项式插值
            # s(t) = 10t^3 - 15t^4 + 6t^5
            position_ratio = 10 * s**3 - 15 * s**4 + 6 * s**5
            velocity_ratio = (30 * s**2 - 60 * s**3 + 30 * s**4) / duration
            acceleration_ratio = (60 * s - 180 * s**2 + 120 * s**3) / duration**2

            position = start_state.positions + (goal_position - start_state.positions) * position_ratio
            velocity = (goal_position - start_state.positions) * velocity_ratio
            acceleration = (goal_position - start_state.positions) * acceleration_ratio

            trajectory.append(TrajectoryPoint(
                position=position,
                velocity=velocity,
                acceleration=acceleration,
                time=t
            ))

        return trajectory


class ImpedanceController:
    """阻抗控制器：柔顺力控制"""

    def __init__(self, stiffness: np.ndarray, damping: np.ndarray,
                 mass: np.ndarray):
        """
        Args:
            stiffness: 刚度系数 [n_dof]
            damping: 阻尼系数 [n_dof]
            mass: 质量系数 [n_dof]
        """
        self.K = np.diag(stiffness)
        self.D = np.diag(damping)
        self.M = np.diag(mass)

    def compute_torque(self,
                       desired_position: np.ndarray,
                       current_position: np.ndarray,
                       current_velocity: np.ndarray,
                       external_force: Optional[np.ndarray] = None) -> np.ndarray:
        """
        计算阻抗控制力矩

        M * x_dd + D * x_d + K * (x - x_d) = F_ext

        Args:
            desired_position: 期望位置
            current_position: 当前位置
            current_velocity: 当前速度
            external_force: 外部力
        Returns:
            控制力矩
        """
        position_error = desired_position - current_position
        torque = (
            self.K @ position_error -
            self.D @ current_velocity
        )

        if external_force is not None:
            torque += external_force

        return torque
```

### 3. 大语言模型驱动的机器人控制

```python
"""
基于大语言模型的机器人高层决策系统
将自然语言指令转化为机器人行为
"""
from typing import List, Dict, Any
from enum import Enum

class RobotAction(Enum):
    """机器人动作类型"""
    MOVE = "move"
    GRASP = "grasp"
    PLACE = "place"
    NAVIGATE = "navigate"
    WAIT = "wait"
    SAY = "say"

class LLMRobotController:
    """LLM 驱动的机器人控制器"""

    def __init__(self, llm_client):
        self.llm = llm_client
        self.available_actions = {
            "move": "移动到指定坐标位置",
            "grasp": "抓取指定物体",
            "place": "将物体放置到指定位置",
            "navigate": "导航到指定区域",
            "wait": "等待",
            "say": "说出指定文本"
        }
        self.object_registry = {}
        self.action_history: List[Dict] = []

    def parse_instruction(self,
                          instruction: str,
                          environment_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        将自然语言指令解析为可执行的机器人动作序列

        Args:
            instruction: 自然语言指令
            environment_state: 当前环境状态
        Returns:
            动作序列
        """
        prompt = f"""你是一个机器人控制规划器。请将以下自然语言指令转换为结构化的机器人动作序列。

可用动作：
{chr(10).join([f"- {k}: {v}" for k, v in self.available_actions.items()])}

当前环境状态：
{environment_state}

用户指令：{instruction}

请输出 JSON 格式的动作序列：
{{
    "actions": [
        {{
            "action": "动作类型",
            "parameters": {{参数}},
            "description": "动作描述"
        }}
    ],
    "reasoning": "规划理由"
}}
"""

        response = self.llm.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2
        )

        import json
        plan = json.loads(response.choices[0].message.content)

        return plan.get("actions", [])

    def execute_action(self, action: Dict[str, Any]) -> bool:
        """
        执行单个机器人动作

        Args:
            action: 动作定义
        Returns:
            是否执行成功
        """
        action_type = action.get("action")
        params = action.get("parameters", {})

        success = False

        if action_type == RobotAction.MOVE.value:
            success = self._execute_move(params)
        elif action_type == RobotAction.GRASP.value:
            success = self._execute_grasp(params)
        elif action_type == RobotAction.PLACE.value:
            success = self._execute_place(params)
        elif action_type == RobotAction.NAVIGATE.value:
            success = self._execute_navigate(params)
        elif action_type == RobotAction.SAY.value:
            success = self._execute_say(params)
        elif action_type == RobotAction.WAIT.value:
            import time
            time.sleep(params.get("duration", 1.0))
            success = True

        # 记录执行历史
        self.action_history.append({
            "action": action,
            "success": success,
            "timestamp": None  # 实际项目中记录时间戳
        })

        return success

    def execute_plan(self,
                     instruction: str,
                     environment_state: Dict[str, Any],
                     max_retries: int = 3) -> Dict[str, Any]:
        """
        执行完整的指令计划

        Args:
            instruction: 用户指令
            environment_state: 环境状态
            max_retries: 最大重试次数
        Returns:
            执行结果
        """
        # 解析指令
        actions = self.parse_instruction(instruction, environment_state)

        results = []
        for i, action in enumerate(actions):
            success = False
            for attempt in range(max_retries):
                success = self.execute_action(action)
                if success:
                    break

                # 失败时的反馈和调整
                if attempt < max_retries - 1:
                    adjusted_action = self._adjust_action_on_failure(action, attempt)
                    action = adjusted_action

            results.append({
                "action": action,
                "success": success,
                "attempts": attempt + 1
            })

            if not success:
                break  # 停止执行

        return {
            "instruction": instruction,
            "results": results,
            "completed": all(r["success"] for r in results)
        }

    def _execute_move(self, params: Dict) -> bool:
        """执行移动动作"""
        # 实际项目中调用机器人控制 API
        target = params.get("target", [0, 0, 0])
        print(f"移动到: {target}")
        return True

    def _execute_grasp(self, params: Dict) -> bool:
        """执行抓取动作"""
        target = params.get("object", "未知物体")
        print(f"抓取: {target}")
        return True

    def _execute_place(self, params: Dict) -> bool:
        """执行放置动作"""
        target = params.get("location", "未知位置")
        print(f"放置到: {target}")
        return True

    def _execute_navigate(self, params: Dict) -> bool:
        """执行导航动作"""
        destination = params.get("destination", "未知目的地")
        print(f"导航到: {destination}")
        return True

    def _execute_say(self, params: Dict) -> bool:
        """执行语音输出"""
        text = params.get("text", "")
        print(f"说: {text}")
        return True

    def _adjust_action_on_failure(self,
                                   action: Dict[str, Any],
                                   attempt: int) -> Dict[str, Any]:
        """失败时调整动作参数"""
        # 实际项目中应基于传感器反馈进行自适应调整
        return action
```

## Sim-to-Real 迁移

具身智能开发中的一大挑战是将仿真环境中训练的模型迁移到真实机器人上：

| 迁移策略 | 原理 | 优势 | 局限 |
|---------|------|------|------|
| 域随机化 | 在仿真中随机化物理参数 | 增强模型鲁棒性 | 可能过度保守 |
| 域适应 | 训练域不变的特征表示 | 保留仿真知识 | 需要真实数据 |
| 元学习 | 学习快速适应新环境 | 在线适应能力强 | 训练复杂 |
| 系统辨识 | 校准仿真器参数 | 缩小仿真-真实差距 | 需要真实测量 |
| 在线微调 | 在真实环境中微调 | 直接优化 | 有风险、成本高 |

## 总结

具身智能将 AI 从纯数字世界带入物理世界，是 AI 发展的下一个重要阶段。机器人感知系统融合多模态传感器数据来理解环境，运动规划与控制系统确保精确、安全的物理操作，大语言模型则为机器人提供了高层语义理解和自主决策能力。Sim-to-Real 迁移技术是连接仿真训练与真实部署的关键桥梁。随着硬件和算法的同步进步，具身智能将在家庭服务、工业制造、医疗和自动驾驶等领域发挥越来越重要的作用。

---

**下一篇**: [99. AI 安全](./99-ai-safety.md)
