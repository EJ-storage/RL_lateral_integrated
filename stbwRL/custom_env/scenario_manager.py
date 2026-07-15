from typing import Optional
import math

import numpy as np

from .datatype.scenario import Scenario


class ScenarioManager:
    def __init__(self, config, rng: Optional[np.random.Generator] = None) -> None:
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng()
        self.current_scenario: Optional[Scenario] = None
        self.last_reward: Optional[float] = None
        self.last_terminated: bool = False
        self.last_truncated: bool = False
        self.last_info = {}
 
    def sample_new_scenario(self) -> Scenario: # 시나리오 샘플링
        scenario = Scenario.sample(self.config, self.rng)
        self.current_scenario = scenario
        return scenario
    
    def reset(self) -> Scenario:
        reward_last = self.last_reward

        make_new = self.current_scenario is None # 시나리오가 생성 조건

        if reward_last is None: # 직전 에피소드 보상이 존재하지 않으면
            make_new = True # 시나리오 생성
        elif not math.isfinite(float(reward_last)): # 보상이 유효하지 않으면
            make_new = True # 시나리오 생성
        elif float(reward_last) > 0.0: # 직전 에피소드 보상이 양수인지
            make_new = True # 시나리오 생성

        if make_new:
            scenario = self.sample_new_scenario()
        else: # 리워드가 정상적이고 0 이하인 경우 현재 시나리오의 steering_max를 0.99배로 줄임
            scenario = self.current_scenario.with_steering_max(
                self.current_scenario.steering_max * 0.99
            )
            self.current_scenario = scenario

        scenario.reset_reference_history() # 이전 에피소드 목표 곡률 이어받지 않도록 초기화
        return scenario

    def update_after_episode(self, reward, terminated: bool, truncated: bool, info) -> None:
        self.last_reward = float(reward)
        self.last_terminated = bool(terminated)
        self.last_truncated = bool(truncated)
        self.last_info = dict(info or {})
