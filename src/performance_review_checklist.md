# 성능이 비슷하게 나올 때 점검 체크리스트 (실행 방법 포함)

아래는 "무엇을 볼지"가 아니라 "어떻게 점검할지" 중심으로 정리한 절차입니다.

---

## 0) 먼저 결과 파일 위치 확인
- 요약: `src/out_bev_ranges/rb_simulation_summary.csv`
- (있다면) 슬롯/사용자 디버그: 시뮬레이터가 저장하는 debug csv

핵심은 **scheduler별로 같은 조건에서 비교**하는 것입니다.

---

## 1) 포화 여부 점검 (가장 먼저)

### 무엇을 확인?
- `mean_requested_rb_per_slot` vs `mean_allocated_rb_per_slot`
- `mean_unserved_users_per_slot`

### 어떻게 확인?
1. summary csv를 열어서 scheduler별 값을 나란히 둡니다.
2. 아래 둘 중 하나면 포화 가능성이 큼:
   - `requested >> allocated`
   - `unserved`가 지속적으로 큼

### 해석 기준(실무용)
- `allocated`가 거의 고정 상한(예: 12) 근처 + `unserved` 큼
  - → 정책이 달라도 모두 "줄 세우기" 상태라 평균 성능이 비슷해지기 쉽습니다.

---

## 2) 공통 병목이 알고리즘 차이를 덮는지 점검

### 무엇을 확인?
- 활성 사용자 수(`mean_active_users_per_slot`)가 scheduler 간 거의 동일한지
- 자원 총량(`total_rb`)과 트래픽 강도가 너무 빡빡한지

### 어떻게 확인?
1. scheduler별 `mean_active_users_per_slot` 비교
2. 동일 시나리오에서 `requested/allocated` 비율 비교
3. 비율이 모두 매우 높으면 "공통 병목"으로 판단

### 추가 실험(필수)
- 부하를 낮춘 설정(사용자 수↓ 또는 도착률↓)으로 1회 재실행
- 이때 성능 차이가 벌어지면, 기존 결과는 병목 지배라고 볼 수 있습니다.

---

## 3) 구현상 동질화(사실상 비슷한 동작) 점검

### 무엇을 확인?
- scheduler 분기 전 공통 로직
- 분기 후에도 결국 비슷한 사용자에게 RB가 몰리는지

### 코드에서 볼 위치
- 공통 요청량 계산: `requested_rb` 생성 루프
- scheduler 분기: `RR / MaxThroughput / PF / Ours / OursPF`
- 공통 집계: `active_users`, `requested_total`, `allocated_total`, `unserved_users`

### 어떻게 확인?
1. debug csv에서 슬롯 단위로 사용자별 `allocated_rb`를 비교
2. 서로 다른 scheduler인데도 반복적으로 같은 사용자만 선택되면 동질화 의심
3. PF 계열은 `avg_thr` 업데이트(이동평균) 민감도도 함께 확인

---

## 4) 지표 민감도 점검 (평균만 보면 안 보임)

### 무엇을 확인?
- 평균 throughput/delay/fairness 외에 분포 지표

### 어떻게 확인?
아래 지표를 추가 집계해 scheduler별 비교:
- 사용자 지연 P95/P99
- 연속 미할당 슬롯 길이(사용자별 max/평균)
- 상태별 처리율(Con/Normal/Empty) 분포

### 해석 포인트
- 평균은 비슷해도 tail(P95/P99)에서 정책 차이가 크게 날 수 있습니다.

---

## 5) 실험 설계 점검

### 무엇을 확인?
- 표본 수 부족/seed 편향 여부

### 어떻게 확인?
1. `n_runs` 증가 (예: 5 → 30 이상)
2. 평균만 보지 말고 표준편차/표준오차도 같이 기록
3. seed 고정 비교(정책 차이만 확인)와 seed 변경 비교(일반화) 분리

---

## 6) 바로 따라할 4단계 실행 순서

1. **summary에서 포화 확인**: `requested/allocated/unserved`를 scheduler별 비교
2. **debug에서 선택 편향 확인**: 슬롯별 `allocated_rb`, `selected_flag` 확인
3. **비포화 조건 재실험**: 사용자 수/도착률 낮춰 재실행
4. **평균+tail 같이 판단**: throughput 평균 + 지연 P95/P99 + 연속 미할당 길이

---

## 7) 빠른 판정 규칙 (의사결정용)
- 아래 3개가 동시에 보이면 "병목 지배로 인한 유사 성능" 가능성이 높음:
  1) `requested >> allocated`
  2) `allocated`가 상한 근처에서 고정
  3) `unserved`가 높게 유지

이 경우 먼저 알고리즘 미세튜닝보다 **부하 구간 분리(비포화/포화)** 실험을 우선하세요.

---

## 8) 제공해주신 표에 바로 적용한 해석 예시

아래 값은 질문에 포함된 표 기준입니다.

### 8-1. 포화 판정
- `allocated = 12`가 모든 scheduler/상태에서 동일
- `active ≈ 60`, `unserved ≈ 48`
  - 즉, 평균적으로 활성 사용자 60명 중 48명이 슬롯에서 미할당
- `requested/allocated` 비율도 매우 큼
  - Empty-MaxThroughput: `819/12 ≈ 68.3`
  - Empty-Ours: `511/12 ≈ 42.6`
  - Normal-Ours: `1364/12 ≈ 113.7`

=> 강한 포화 구간이므로, 정책 차이가 평균치에서 축소되는 조건이 맞습니다.

### 8-2. 그런데 이번 표는 "완전히 비슷"하지는 않음
- Empty 상태:
  - Ours throughput `65.07M` vs RR/PF/OursPF `~63.8~63.9M`
  - Ours delay `1.71s` vs RR/PF/OursPF `~2.0s`
- Normal 상태:
  - Ours throughput `63.59M` vs RR/PF/OursPF `~62.0~62.2M`
  - Ours delay `19.09s` vs RR/PF/OursPF `~19.84~19.95s`

=> 포화 상황에서도 **Ours가 일관되게 우세**합니다(특히 MT 대비 격차 큼).

### 8-3. 다음 확인 포인트(이 표 기준)
1. debug csv에서 Ours가 어떤 사용자군(채널/상태)에 RB를 더 주는지 확인
2. 사용자별 지연 P95/P99, 연속 미할당 길이 비교
3. 비포화 세팅(예: active 20~30 수준)에서 재실행해 정책 고유 효과 분리

이 3가지를 보면 "왜 Ours가 포화에서도 이득이 나는지"를 더 명확히 설명할 수 있습니다.

## PR 작성 언어
- 이 저장소 관련 PR 제목/본문은 한국어로 작성합니다.
