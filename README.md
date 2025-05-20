# 2025 무인항공딥러닝개론

## 📚 Citing RESCO

This project is based on **RESCO** and was used in the context of _Reinforcement Learning Benchmarks for Traffic Signal Control_.  
If you use RESCO in your work, please include the following citation:

```bibtex
@inproceedings{ault2021reinforcement,
  title={Reinforcement Learning Benchmarks for Traffic Signal Control},
  author={James Ault and Guni Sharon},
  booktitle={Proceedings of the Thirty-fifth Conference on Neural Information Processing Systems (NeurIPS 2021) Datasets and Benchmarks Track},
  month={December},
  year={2021}
} 
```

## ⚙️ Modifications for Experiment

- 본 실험은 위 논문에서 제안된 **RESCO 시나리오**를 기반으로 수행되었습니다.
- 사용된 강화학습 프레임워크는 **[PFRL](https://github.com/pfnet/pfrl)**이며, 내부의 `agent.py` 모듈에서  
  **`tuple` 형태로 반환되던 값을 `list` 형태로 처리할 수 있도록 수정**하였습니다.
- 시뮬레이션 전체의 **평균 속도 (Average Speed)** 와 **평균 대기 시간 (Average Waiting Time)** 을 측정하기 위해  
  `sumo cmd`의 입력 인자를 수정하였습니다.
- 이는 **단일 교차로가 아닌 네트워크 전체 수준에서의 평가를 가능하게 하기 위한 조치**입니다.
