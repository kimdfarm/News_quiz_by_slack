## 대상 문서 / 데이터

* **무엇을:** IT 및 경제 분야 주요 뉴스 기사 텍스트 및 RSS 피드
* **몇 건:** 일일 10건 내외 (1주일 유지 시 최대 약 70건 스토리지 관리)
* **형식:** JSON / Markdown / SQL 관계형 데이터 (`pgvector` 또는 Neon/PostgreSQL 활용)

## 분할·메타데이터 설계 ★ (2교시 2절 반영)

* **분할 방식:** 구조 기반(Section/Paragraph split) → 뉴스의 본문과 저자 의견/사설 영역을 분리하여 청크화
* **chunk_size / overlap:** `chunk_size: 500`, `overlap: 50`
* **싱글 메타데이터:** `source` (뉴스 출처), `page` (날짜/일련번호), `author_opinion` (저자 의견 유무), `category` (경제/IT)

## 처리 흐름

슬랙 질문 (`@봇 이름 문제 내줘`) $\rightarrow$ DB 검색 및 1주일 이내 최신 뉴스 필터링 $\rightarrow$ RAG/LLM(Gemma 3.4b) 프롬프트 템플릿 주입 및 퀴즈 생성 $\rightarrow$ 슬랙 스레드 답변

## 범위 밖 (안 하는 것) ★

* 네이버/다음 등 대형 포털 사이트의 전체 실시간 대규모 크롤링 (차단 위험 및 과제 범위 초과 방지)
* 12B 이상의 대형 상용 모델 직접 구동 (VRAM 초과 방지)
* 1주일이 지난 뉴스 데이터의 영구 보관 (DB 용량 관리 및 슬라이딩 윈도우 TTL 정책 준수)
