import os
import streamlit as st
import pandas as pd
from io import BytesIO
import json
import time
import os

# LangChain 관련 라이브러리 임포트
from langchain.prompts import ChatPromptTemplate
from langchain_unstructured import UnstructuredLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.schema import AIMessage, HumanMessage, SystemMessage


# 폴더가 없으면 생성
documents_path = "./documents/"
if not os.path.exists(documents_path):
    os.makedirs(documents_path)

###############################################################################
# 0. OpenAI 클라이언트 초기화 & 시스템 프롬프트
###############################################################################
# API_KEY를 환경 변수에서 가져오기
OPENAI_API_KEY = st.secrets["openai"]["api_key"]

if not OPENAI_API_KEY:
    st.error("OpenAI API 키가 설정되지 않았습니다. 환경 변수를 확인하세요.")
    st.stop()

SYSTEM_PROMPT = """한국의 초등학교 2022 개정 교육과정 전문가입니다.
학교자율시간 계획서를 다음 원칙에 따라 작성합니다:

1. 지도계획에 모든 차시에 학습내용과 학습 주제가 빈틈없이 내용이 꼭 들어가야 합니다.
2. 학습자 중심의 교육과정 구성
3. 실생활 연계 및 체험 중심 활동
4. 교과 간 연계 및 통합적 접근
5. 과정 중심 평가와 피드백 강조
6. 유의미한 학습경험 제공
7. 요구사항을 반영한 맞춤형 교육과정 구성
8. 교수학습 방법의 다양화
9. 객관적이고 공정한 평가계획 수립
"""

###############################################################################
# 1. 페이지 기본 설정
###############################################################################
def set_page_config():
    """페이지 기본 설정"""
    try:
        st.set_page_config(page_title="학교자율시간 계획서 생성기", page_icon="📚", layout="wide")
    except:
        pass

    st.markdown("""
        <style>
        .main .block-container { 
            padding: 2rem; 
            max-width: 1200px; 
        }
        .step-header {
            background-color: #f8fafc;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .edit-container {
            border: 1px solid #e2e8f0;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .stButton > button {
            background-color: #3b82f6;
            color: white;
            border-radius: 0.5rem;
            padding: 0.5rem 1rem;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #2563eb;
            box-shadow: 0 4px 6px rgba(37, 99, 235, 0.2);
            transform: translateY(-1px);
        }
        .stProgress > div > div > div {
            background-color: #3b82f6;
        }
        </style>
    """, unsafe_allow_html=True)

###############################################################################
# 2. 진행 상황 표시
###############################################################################
def show_progress():
    """진행 상황 표시"""
    current_step = st.session_state.get('step', 1)
    steps = ["기본정보", "목표/내용", "성취기준", "교수학습/평가", "차시별계획", "최종 검토"]

    st.markdown("""
        <style>
        .step-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 1rem 0;
            position: relative;
            flex-direction: row;
            width: 100%;
            padding: 20px;
        }
        .step-item {
            display: flex;
            flex-direction: column;
            align-items: center;
            flex: 1;
            position: relative;
            z-index: 2;
        }
        .step-circle {
            width: 50px;
            height: 50px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            position: relative;
            margin-bottom: 8px;
        }
        .step-active {
            background-color: #3b82f6;
            color: white;
            box-shadow: 0 0 10px rgba(59, 130, 246, 0.5);
            transform: scale(1.1);
            transition: all 0.3s ease;
        }
        .step-completed {
            background-color: #10b981;
            color: white;
        }
        .step-pending {
            background-color: #e5e7eb;
            color: #6b7280;
        }
        .step-label {
            font-size: 0.9rem;
            color: #374151;
            text-align: center;
            margin-top: 4px;
        }
        .step-line {
            height: 4px;
            flex: 1;
            background-color: #e5e7eb;
            margin: 0 10px;
            position: relative;
            top: -25px;
            z-index: 1;
        }
        .step-line-completed {
            background-color: #10b981;
        }
        .step-line-active {
            background-color: #3b82f6;
        }
        .step-container-outer {
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
            position: relative;
            padding: 10px 20px;
        }
        </style>
    """, unsafe_allow_html=True)

    html = '<div class="step-container-outer"><div class="step-container">'

    for i, step in enumerate(steps, 1):
        if i < current_step:
            circle_class = "step-completed"
            icon = "✓"
            line_class = "step-line-completed"
        elif i == current_step:
            circle_class = "step-active"
            icon = str(i)
            line_class = "step-line-active"
        else:
            circle_class = "step-pending"
            icon = str(i)
            line_class = "step-line-pending"

        html += f'''
            <div class="step-item">
                <div class="step-circle {circle_class}">{icon}</div>
                <div class="step-label">{step}</div>
            </div>
        '''

        if i < len(steps):
            if i < current_step:
                line_style = "step-line-completed"
            elif i == current_step:
                line_style = "step-line-active"
            else:
                line_style = "step-line-pending"
            html += f'<div class="step-line {line_style}"></div>'

    html += '</div></div>'
    st.markdown(html, unsafe_allow_html=True)

###############################################################################
# 3. 벡터 데이터베이스 설정
###############################################################################
@st.cache_resource(show_spinner="문서를 임베딩하는 중...")
def setup_vector_store():
    """문서를 로드하고 벡터 스토어를 설정합니다."""
    try:
        documents_dir = "./documents/"
        supported_extensions = ["pdf", "txt", "docx"]

        all_docs = []
        for filename in os.listdir(documents_dir):
            if any(filename.lower().endswith(ext) for ext in supported_extensions):
                file_path = os.path.join(documents_dir, filename)
                loader = UnstructuredLoader(file_path)
                documents = loader.load()
                all_docs.extend(documents)

        if not all_docs:
            st.error("`documents/` 폴더에 문서가 없습니다.")
            return None

        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vector_store = FAISS.from_documents(all_docs, embeddings)
        st.success("벡터 스토어가 성공적으로 생성되었습니다.")

        return vector_store

    except Exception as e:
        st.error(f"벡터 스토어 설정 중 오류가 발생했습니다: {str(e)}")
        return None

###############################################################################
# 4. OpenAI 호출 함수
###############################################################################
def generate_content(step, data, vector_store):
    """단계별 안내 메시지를 만들고 LangChain을 통해 JSON을 생성 후 파싱"""
    try:
        context = ""
        if step > 1 and vector_store:
            # RAG를 통해 관련 문서 검색
            retriever = vector_store.as_retriever()
            query = {
                2: "목표와 내용 요소",
                3: "성취기준",
                4: "교수학습 방법과 평가계획"
            }.get(step, "")

            if query:
                retrieved_docs = retriever.get_relevant_documents(query)
                context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # 단계별 프롬프트 정의
        step_prompts = {
            1: f"""학교자율시간 활동의 기본 정보를 작성해주세요.
아래 사항을 고려하여 JSON 형식으로 기술합니다.
1) 학교 교육목표 및 비전과 어떻게 연계되는지 강조
2) 학교 및 학생의 요구(학습자 특성, 지역사회 자원 등)를 반영한 활동 필요성 구체적으로
3) 활동 개요에 대상 학년, 총 시수, 주간 운영 시수, 활동 형식(프로젝트, 탐구학습, 토론 등) 간단히 포함
4) 교육적 의의, 운영 방향(지향 교수학습방법, 학생 참여방식 등) 명시

활동명: {data.get('activity_name')}
요구사항: {data.get('requirements')}
학교급: {data.get('school_type')}
대상 학년: {', '.join(data.get('grades', []))}
연계 교과: {', '.join(data.get('subjects', []))}
총 차시: {data.get('total_hours')}차시
주당 차시: {data.get('weekly_hours')}차시
운영 학기: {', '.join(data.get('semester', []))}

다음 JSON 형식으로 작성:
{{
    "necessity": "(학교 비전, 학습자 요구, 지역사회 연계 가능성 등을 포함)",
    "overview": "(대상 학년, 총 시수, 활동 형식 등을 포함)",
    "characteristics": "(교육적 의의, 운영 방향, 교수학습 전략 등)"
}}""",

            2: f"""{context}

위 정보를 바탕으로 학교자율시간 활동의 목표와 주요 내용 요소를 정리해주세요.
1) 지식, 기능, 태도 영역으로 구분된 목표 작성
2) 활동이 다루는 영역(단일/통합 교과 등) 기술
3) 핵심 개념 또는 원리 2~3개 이상 제시, 간단 설명 가능

활동명: {data.get('activity_name')}
요구사항: {data.get('requirements')}

다음 JSON 형식으로 작성:
{{
    "goals": [
        "(지식 영역 목표)",
        "(기능 영역 목표)",
        "(태도 영역 목표)"
    ],
    "domain": "(단일 또는 통합 영역 명시)",
    "key_ideas": [
        "(핵심 개념/원리1)",
        "(핵심 개념/원리2)",
        "(핵심 개념/원리3)"
    ]
}}""",

            3: f"""{context}

위 정보를 종합하여 학교자율시간 활동의 성취기준을 작성해주세요.
1) 성취기준 코드는 고유하게 (예: '3사코딩_01')
2) 성취기준 설명은 학생들이 달성해야 할 학습 결과를 구체적으로
3) A/B/C 수준 구분, 각 수준의 구체적 성취 모습을 간단히

활동명: {data.get('activity_name')}
영역: {data.get('domain')}

다음 JSON 형식으로 작성:
[
    {{
        "code": "(성취기준 코드)",
        "description": "(성취기준 설명)",
        "levels": [
            {{"level": "A", "description": "(A수준 성취기준)"}},
            {{"level": "B", "description": "(B수준 성취기준)"}},
            {{"level": "C", "description": "(C수준 성취기준)"}}
        ]
    }}
]""",

            4: f"""{context}

위 정보를 바탕으로 학교자율시간 활동의 교수학습 방법과 평가계획을 작성해주세요.
1) 교수학습 방법: 학생 중심의 다양한 교수학습 방법을 구체적으로
2) 평가계획: 과정 중심 평가 및 구체적인 평가 방법 상세 기술

활동명: {data.get('activity_name')}
요구사항: {data.get('requirements')}

다음 JSON 형식으로 작성:
{{
    "teaching_methods": [
        {{"method": "프로젝트 기반 학습", "description": "학생들이 직접 프로젝트를 기획하고 실행함으로써 문제 해결 능력을 기릅니다."}},
        {{"method": "토론 활동", "description": "학생들이 다양한 주제에 대해 토론함으로써 의사소통 능력을 향상시킵니다."}}
    ],
    "assessment_plan": [
        {{"focus": "과정 중심 평가", "description": "학생들의 학습 과정과 참여도를 평가합니다."}},
        {{"focus": "형성 평가", "description": "수업 중간에 학생들의 이해도를 점검하고 피드백을 제공합니다."}}
    ]
}}"""
        }

        if step == 5:
            return {}

        prompt = step_prompts.get(step, "")
        if prompt:
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ]

            # LangChain의 ChatOpenAI를 사용하여 응답 생성
            chat = ChatOpenAI(
                openai_api_key=OPENAI_API_KEY,
                model="gpt-4o",  # 모델 이름 오타 수정
                temperature=0.7,
                max_tokens=2048
            )

            response = chat(messages)
            content = response.content.strip()
            content = content.replace('```json', '').replace('```', '').strip()

            try:
                parsed = json.loads(content)
                if step == 4:
                    # 데이터 구조 검증
                    if 'teaching_methods' in parsed and 'assessment_plan' in parsed:
                        for method in parsed['teaching_methods']:
                            if not isinstance(method, dict) or 'method' not in method or 'description' not in method:
                                raise ValueError("Invalid structure in teaching_methods")
                        for assessment in parsed['assessment_plan']:
                            if not isinstance(assessment, dict) or 'focus' not in assessment or 'description' not in assessment:
                                raise ValueError("Invalid structure in assessment_plan")
                return parsed
            except json.JSONDecodeError as e:
                st.warning(f"JSON 파싱 오류가 발생했습니다. 기본값을 사용합니다. 오류: {str(e)}")
                return get_default_content(step)
            except ValueError as ve:
                st.warning(f"데이터 구조 오류: {str(ve)}. 기본값을 사용합니다.")
                return get_default_content(step)

    except Exception as e:
        st.error(f"내용 생성 중 오류가 발생했습니다: {str(e)}")
        return get_default_content(step)

def get_default_content(step):
    """단계별 기본 내용을 반환하는 함수"""
    defaults = {
        1: {
            "necessity": "예시: 2022개정교육과정, 학생/학부모 요구, 지역 여건 부합 등",
            "overview": "예시: 대상 학년, 총 시수, 운영 형태 등",
            "characteristics": "예시: 교육적 의의, 학생 참여 중심 운영 방식 등"
        },
        2: {
            "goals": [
                "지식 영역 목표 예시",
                "기능 영역 목표 예시",
                "태도 영역 목표 예시"
            ],
            "domain": "예: 통합 교과(사회+과학 등)",
            "key_ideas": [
                "예: 프로젝트 학습을 통한 문제 해결 능력 신장",
                "예: 토론 활동을 통한 의사소통 역량 강화"
            ]
        },
        3: [{
            "code": "SD_01",
            "description": "학생이 자료를 수집, 분석하여 문제 정의와 해결 방안을 제시할 수 있다.",
            "levels": [
                {"level": "A", "description": "다양한 자료를 능동적으로 활용해 창의적인 해결 방안을 설계한다."},
                {"level": "B", "description": "제시된 자료를 활용해 해결 방안을 세운다."},
                {"level": "C", "description": "주어진 자료를 활용해 해결 방안 초안을 구성한다."}
            ]
        }],
        4: {
            "teaching_methods": [
                {"method": "프로젝트 기반 학습", "description": "학생들이 직접 프로젝트를 기획하고 실행함으로써 문제 해결 능력을 기릅니다."},
                {"method": "토론 활동", "description": "학생들이 다양한 주제에 대해 토론함으로써 의사소통 능력을 향상시킵니다."}
            ],
            "assessment_plan": [
                {"focus": "과정 중심 평가", "description": "학생들의 학습 과정과 참여도를 평가합니다."},
                {"focus": "형성 평가", "description": "수업 중간에 학생들의 이해도를 점검하고 피드백을 제공합니다."}
            ]
        }
    }
    return defaults.get(step, {})

###############################################################################
# 5. 단계별 UI 함수
###############################################################################
def show_step_1(vector_store):
    """1단계: 기본 정보 입력 및 생성"""
    st.markdown("<div class='step-header'><h3>1단계: 기본 정보</h3></div>", unsafe_allow_html=True)

    if 'generated_step_1' not in st.session_state:
        # 데이터 입력 및 생성 단계
        with st.form("basic_info_form"):
            school_type = st.radio("학교급", ["초등학교", "중학교"], horizontal=True, key="school_type_radio")

            col1, col2 = st.columns(2)
            with col1:
                total_hours = st.number_input(
                    "총 차시",
                    min_value=1,
                    max_value=68,
                    value=st.session_state.data.get('total_hours', 34),
                    help="학교자율시간의 총 차시를 입력하세요 (최대 68차시)"
                )

                weekly_hours = st.number_input(
                    "주당 차시",
                    min_value=1,
                    max_value=2,
                    value=st.session_state.data.get('weekly_hours', 1),
                    help="주당 수업 차시를 입력하세요"
                )

            with col2:
                semester = st.multiselect(
                    "운영 학기",
                    ["1학기", "2학기"],
                    default=st.session_state.data.get('semester', ["1학기"])
                )

            st.markdown("#### 학년 선택")
            if school_type == "초등학교":
                grades = st.multiselect(
                    "학년",
                    ["3학년", "4학년", "5학년", "6학년"],
                    default=st.session_state.data.get('grades', []),
                    key="grades_multiselect_elem"
                )
                subjects = st.multiselect(
                    "교과",
                    ["국어", "수학", "사회", "과학", "영어", "음악", "미술", "체육", "실과", "도덕"],
                    default=st.session_state.data.get('subjects', []),
                    key="subjects_multiselect_elem"
                )
            else:  # 중학교
                grades = st.multiselect(
                    "학년",
                    ["1학년", "2학년", "3학년"],
                    default=st.session_state.data.get('grades', []),
                    key="grades_multiselect_middle"
                )
                subjects = st.multiselect(
                    "교과",
                    ["국어", "수학", "사회/역사", "과학/기술", "영어", "음악", "미술", "체육", "정보", "도덕"],
                    default=st.session_state.data.get('subjects', []),
                    key="subjects_multiselect_middle"
                )

            col1, col2 = st.columns(2)
            with col1:
                activity_name = st.text_input(
                    "활동명",
                    value=st.session_state.data.get('activity_name', ''),
                    placeholder="예: 인공지능 놀이터"
                )
            with col2:
                requirements = st.text_area(
                    "요구사항",
                    value=st.session_state.data.get('requirements', ''),
                    placeholder="예: 학생들의 디지털 리터러시 역량 강화가 필요함",
                    height=100
                )

            # 수정 및 다음 단계 버튼
            submit_button = st.form_submit_button("정보 생성 및 다음 단계로", use_container_width=True)

        # 버튼 동작 처리
        if submit_button:
            if activity_name and requirements and grades and subjects and semester:
                with st.spinner("정보를 생성하고 있습니다..."):
                    # 데이터 저장
                    st.session_state.data.update({
                        'school_type': school_type,
                        'grades': grades,
                        'subjects': subjects,
                        'activity_name': activity_name,
                        'requirements': requirements,
                        'total_hours': total_hours,
                        'weekly_hours': weekly_hours,
                        'semester': semester
                    })

                    # 기본 정보 생성
                    basic_info = generate_content(1, st.session_state.data, vector_store)
                    if basic_info:
                        st.session_state.data.update(basic_info)
                        st.success("기본 정보가 생성되었습니다.")
                        st.session_state.generated_step_1 = True
            else:
                st.error("모든 필수 항목을 입력해주세요.")

    # 생성된 내용 수정 단계
    if 'generated_step_1' in st.session_state:
        with st.form("edit_basic_info_form"):
            st.markdown("#### 생성된 내용 수정")

            necessity = st.text_area(
                "활동의 필요성",
                value=st.session_state.data.get('necessity', ''),
                height=150,
                key="necessity_textarea"
            )
            overview = st.text_area(
                "활동 개요",
                value=st.session_state.data.get('overview', ''),
                height=150,
                key="overview_textarea"
            )
            characteristics = st.text_area(
                "활동의 성격",
                value=st.session_state.data.get('characteristics', ''),
                height=150,
                key="characteristics_textarea"
            )

            # 수정 및 다음 단계 버튼
            submit_button_edit = st.form_submit_button("수정사항 저장 및 다음 단계로", use_container_width=True)

        if submit_button_edit:
            with st.spinner("수정사항을 저장하고 다음 단계로 이동 중입니다..."):
                # 데이터 업데이트
                st.session_state.data.update({
                    'necessity': necessity,
                    'overview': overview,
                    'characteristics': characteristics
                })

                # 수정 완료 플래그 제거
                del st.session_state.generated_step_1

                st.success("수정사항이 저장되었습니다.")
                st.session_state.step = 2
                st.rerun()

    return False

def show_step_2(vector_store):
    """2단계: 목표와 내용 요소 입력 및 생성"""
    st.markdown("<div class='step-header'><h3>2단계: 목표와 내용 요소</h3></div>", unsafe_allow_html=True)

    if 'generated_step_2' not in st.session_state:
        # 데이터 입력 및 생성 단계
        with st.form("goals_content_form"):
            st.info("목표와 내용 요소를 생성합니다.")

            submit_button = st.form_submit_button("목표/내용 생성 및 다음 단계로", use_container_width=True)

        # 버튼 동작 처리
        if submit_button:
            with st.spinner("목표와 내용을 생성하고 있습니다..."):
                # 목표 및 내용 생성
                content = generate_content(2, st.session_state.data, vector_store)
                if content:
                    st.session_state.data.update(content)
                    st.success("목표와 내용이 생성되었습니다.")
                    st.session_state.generated_step_2 = True
    else:
        # 생성된 내용 전체 수정 단계
        with st.form("edit_goals_content_form"):
            st.markdown("#### 생성된 내용 수정")

            # 목표 수정
            st.markdown("##### 목표")
            goals = []
            for i, goal in enumerate(st.session_state.data.get('goals', [])):
                goal_text = st.text_input(
                    f"목표 {i+1}",
                    value=goal,
                    key=f"goal_{i}",
                    help="지식, 기능, 태도 영역별 목표를 작성하세요."
                )
                goals.append(goal_text)

            # 활동 영역 수정
            st.markdown("##### 활동 영역")
            domain = st.text_input(
                "활동 영역",
                value=st.session_state.data.get('domain', ''),
                key="domain_input",
                help="단일 교과 또는 통합 교과 영역을 입력하세요."
            )

            # 핵심 아이디어 수정
            st.markdown("##### 핵심 아이디어")
            key_ideas = []
            for i, idea in enumerate(st.session_state.data.get('key_ideas', [])):
                idea_text = st.text_input(
                    f"핵심 아이디어 {i+1}",
                    value=idea,
                    key=f"idea_{i}",
                    help="주요 개념이나 원리를 입력하세요."
                )
                key_ideas.append(idea_text)

            # 수정 및 다음 단계 버튼
            submit_button_edit = st.form_submit_button("수정사항 저장 및 다음 단계로", use_container_width=True)

        if submit_button_edit:
            with st.spinner("수정사항을 저장하고 다음 단계로 이동 중입니다..."):
                # 데이터 업데이트
                st.session_state.data.update({
                    'goals': goals,
                    'domain': domain,
                    'key_ideas': key_ideas
                })

                # 수정 완료 플래그 제거
                del st.session_state.generated_step_2

                st.success("수정사항이 저장되었습니다.")
                st.session_state.step = 3
                st.rerun()

    return False

def show_step_3(vector_store):
    """3단계: 성취기준 설정 입력 및 생성"""
    st.markdown("<div class='step-header'><h3>3단계: 성취기준 설정</h3></div>", unsafe_allow_html=True)

    if 'generated_step_3' not in st.session_state:
        # 데이터 입력 및 생성 단계
        with st.form("standards_form"):
            st.info("성취기준을 생성합니다.")

            submit_button = st.form_submit_button("성취기준 생성 및 다음 단계로", use_container_width=True)

        # 버튼 동작 처리
        if submit_button:
            with st.spinner("성취기준을 생성하고 있습니다..."):
                # 성취기준 생성
                standards = generate_content(3, st.session_state.data, vector_store)
                if standards:
                    st.session_state.data['standards'] = standards
                    st.success("성취기준이 생성되었습니다.")
                    st.session_state.generated_step_3 = True
    else:
        # 생성된 성취기준 전체 수정 단계
        with st.form("edit_standards_form"):
            st.markdown("#### 생성된 성취기준 수정")

            edited_standards = []
            for i, standard in enumerate(st.session_state.data.get('standards', [])):
                st.markdown(f"##### 성취기준 {i+1}")
                code = st.text_input(
                    "성취기준 코드",
                    value=standard['code'],
                    key=f"std_code_{i}",
                    help="예: 3사코딩_01"
                )
                description = st.text_area(
                    "성취기준 설명",
                    value=standard['description'],
                    key=f"std_desc_{i}",
                    height=100,
                    help="학생들이 달성해야 할 구체적인 학습 결과를 작성하세요."
                )

                st.markdown("##### 수준별 성취기준")
                col1, col2, col3 = st.columns(3)

                with col1:
                    a_desc = st.text_area(
                        "A 수준",
                        value=next((l['description'] for l in standard['levels'] if l['level'] == 'A'), ''),
                        key=f"std_{i}_level_A",
                        height=100,
                        help="A 수준 성취기준을 작성하세요."
                    )

                with col2:
                    b_desc = st.text_area(
                        "B 수준",
                        value=next((l['description'] for l in standard['levels'] if l['level'] == 'B'), ''),
                        key=f"std_{i}_level_B",
                        height=100,
                        help="B 수준 성취기준을 작성하세요."
                    )

                with col3:
                    c_desc = st.text_area(
                        "C 수준",
                        value=next((l['description'] for l in standard['levels'] if l['level'] == 'C'), ''),
                        key=f"std_{i}_level_C",
                        height=100,
                        help="C 수준 성취기준을 작성하세요."
                    )

                edited_standards.append({
                    "code": code,
                    "description": description,
                    "levels": [
                        {"level": "A", "description": a_desc},
                        {"level": "B", "description": b_desc},
                        {"level": "C", "description": c_desc}
                    ]
                })
                st.markdown("---")

            # 수정 및 다음 단계 버튼
            submit_button_edit = st.form_submit_button("수정사항 저장 및 다음 단계로", use_container_width=True)

        if submit_button_edit:
            with st.spinner("수정사항을 저장하고 다음 단계로 이동 중입니다..."):
                # 데이터 업데이트
                st.session_state.data['standards'] = edited_standards

                # 수정 완료 플래그 제거
                del st.session_state.generated_step_3

                st.success("성취기준이 저장되었습니다.")
                st.session_state.step = 4
                st.rerun()

    return False

def show_step_4(vector_store):
    """4단계: 교수학습 방법 및 평가계획 입력 및 생성"""
    st.markdown("<div class='step-header'><h3>4단계: 교수학습 방법 및 평가계획</h3></div>", unsafe_allow_html=True)

    if 'generated_step_4' not in st.session_state:
        # 데이터 입력 및 생성 단계
        with st.form("teaching_assessment_form"):
            st.info("교수학습 방법 및 평가계획을 생성합니다.")

            submit_button = st.form_submit_button("교수학습/평가계획 생성 및 다음 단계로", use_container_width=True)

        # 버튼 동작 처리
        if submit_button:
            with st.spinner("교수학습 방법 및 평가계획을 생성하고 있습니다..."):
                # 교수학습 방법 및 평가계획 생성
                content = generate_content(4, st.session_state.data, vector_store)
                if content:
                    st.session_state.data.update({
                        'teaching_methods': content.get('teaching_methods', []),
                        'assessment_plan': content.get('assessment_plan', [])
                    })
                    st.success("교수학습 방법 및 평가계획이 생성되었습니다.")
                    st.session_state.generated_step_4 = True
    else:
        # 생성된 교수학습 방법 및 평가계획 전체 수정 단계
        with st.form("edit_teaching_assessment_form"):
            st.markdown("#### 생성된 교수학습 방법 및 평가계획 수정")

            # 교수학습 방법 수정
            st.markdown("##### 교수학습 방법")
            edited_teaching = []
            for i, method in enumerate(st.session_state.data.get('teaching_methods', [])):
                st.markdown(f"###### 교수학습 방법 {i+1}")
                method_name = st.text_input(
                    "방법",
                    value=method.get('method', ''),
                    key=f"tm_method_{i}",
                    help="교수학습 방법의 이름을 입력하세요."
                )
                method_desc = st.text_area(
                    "설명",
                    value=method.get('description', ''),
                    key=f"tm_desc_{i}",
                    height=80,
                    help="교수학습 방법에 대한 자세한 설명을 입력하세요."
                )
                edited_teaching.append({
                    "method": method_name,
                    "description": method_desc
                })
                st.markdown("---")

            # 평가계획 수정
            st.markdown("##### 평가계획")
            edited_assessment = []
            for i, assessment in enumerate(st.session_state.data.get('assessment_plan', [])):
                st.markdown(f"###### 평가계획 {i+1}")
                focus = st.text_input(
                    "평가 초점",
                    value=assessment.get('focus', ''),
                    key=f"ap_focus_{i}",
                    help="평가의 주요 초점을 입력하세요."
                )
                assessment_desc = st.text_area(
                    "설명",
                    value=assessment.get('description', ''),
                    key=f"ap_desc_{i}",
                    height=80,
                    help="평가 방법에 대한 자세한 설명을 입력하세요."
                )
                edited_assessment.append({
                    "focus": focus,
                    "description": assessment_desc
                })
                st.markdown("---")

            # 수정 및 다음 단계 버튼
            submit_button_edit = st.form_submit_button("수정사항 저장 및 다음 단계로", use_container_width=True)

        if submit_button_edit:
            with st.spinner("수정사항을 저장하고 다음 단계로 이동 중입니다..."):
                # 데이터 업데이트
                st.session_state.data.update({
                    'teaching_methods': edited_teaching,
                    'assessment_plan': edited_assessment
                })

                # 수정 완료 플래그 제거
                del st.session_state.generated_step_4

                st.success("교수학습 방법 및 평가계획이 저장되었습니다.")
                st.session_state.step = 5
                st.rerun()

    return False

###############################################################################
# 6. 차시별 지도계획 생성 함수
###############################################################################
def generate_lesson_plans_in_chunks(total_hours, data, chunk_size=10, vector_store=None):
    """
    chunk_size 단위로 나누어 여러 번 API를 호출하여 lesson_plans를 생성하는 함수.
    예: chunk_size=10 → 한 번에 최대 10차시씩 생성.

    Args:
        total_hours (int): 총 차시 수
        data (dict): 계획서 데이터
        chunk_size (int, optional): 한 번에 생성할 차시 수. 기본값 10
        vector_store: 벡터 스토어 객체

    Returns:
        list: 생성된 차시별 계획 리스트
    """
    all_lesson_plans = []
    progress_bar = st.progress(0)

    try:
        for start in range(0, total_hours, chunk_size):
            end = min(start + chunk_size, total_hours)
            progress = int((start / total_hours) * 100)
            progress_bar.progress(progress)

            st.write(f"{start+1}~{end}차시 계획 생성 중...")

            # 차시별 프롬프트 생성
            chunk_prompt = f"""
다음 정보를 바탕으로 {start+1}차시부터 {end}차시까지의 지도계획을 JSON으로 작성해주세요.

활동명: {data.get('activity_name')}
필요성: {data.get('necessity')}
개요: {data.get('overview')}
성격: {data.get('characteristics')}
목표: {data.get('goals')}
핵심 아이디어: {data.get('key_ideas')}
성취기준: {data.get('standards')}
교수학습 방법: {data.get('teaching_methods')}
평가계획: {data.get('assessment_plan')}

각 차시는 다음 사항을 고려하여 작성해주세요:
1. 차시별로 명확한 학습주제 설정
2. 구체적이고 실천 가능한 학습내용 기술
3. 실제 수업에 필요한 교수학습자료 명시
4. 이전 차시와의 연계성 고려
5. 학습목표 달성을 위한 단계적 구성

다음 JSON 형식으로 작성:
{{
  "lesson_plans": [
    {{
      "lesson_number": "차시번호",
      "topic": "학습주제",
      "content": "학습내용",
      "materials": "교수학습자료"
    }}
  ]
}}
"""

            # LangChain API 호출
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=chunk_prompt)
            ]

            try:
                chat = ChatOpenAI(
                    openai_api_key=OPENAI_API_KEY,
                    model="gpt-4o",  # 모델 이름 오타 수정
                    temperature=0.5,  # 구조적 답변 위해 약간 낮춤
                    max_tokens=2000
                )

                response = chat(messages)
                content = response.content.strip()
                content = content.replace('```json', '').replace('```', '').strip()

                parsed = json.loads(content)
                lesson_plans = parsed.get("lesson_plans", [])

                # 차시 번호 검증 및 수정
                for i, plan in enumerate(lesson_plans, start=start+1):
                    plan["lesson_number"] = str(i)

                all_lesson_plans.extend(lesson_plans)

                # API 호출 제한 방지
                time.sleep(1)

            except json.JSONDecodeError as e:
                st.error(f"{start+1}~{end}차시 생성 중 JSON 파싱 오류 발생: {e}")
                continue
            except Exception as e:
                st.error(f"{start+1}~{end}차시 생성 중 오류 발생: {e}")
                continue

        progress_bar.progress(100)
        return all_lesson_plans

    except Exception as e:
        st.error(f"차시별 계획 생성 중 오류가 발생했습니다: {str(e)}")
        return []

def show_step_5(vector_store):
    """5단계: 차시별 지도계획 입력 및 생성"""
    total_hours = st.session_state.data.get('total_hours', 30)
    st.markdown(f"<div class='step-header'><h3>5단계: 차시별 지도계획 ({total_hours}차시)</h3></div>", unsafe_allow_html=True)

    if 'generated_step_5' not in st.session_state:
        # 데이터 입력 및 생성 단계
        with st.form("lesson_plans_form"):
            st.info(f"{total_hours}차시 계획을 생성합니다.")

            submit_button = st.form_submit_button(f"{total_hours}차시 계획 생성 및 다음 단계로", use_container_width=True)

        # 버튼 동작 처리
        if submit_button:
            with st.spinner(f"{total_hours}차시 계획을 생성하고 있습니다..."):
                chunk_size = 10
                all_plans = generate_lesson_plans_in_chunks(total_hours, st.session_state.data, chunk_size, vector_store)
                if all_plans:
                    st.session_state.data['lesson_plans'] = all_plans
                    st.success(f"{total_hours}차시 계획이 생성되었습니다.")
                    st.session_state.generated_step_5 = True
    else:
        # 생성된 차시별 계획 전체 수정 단계
        with st.form("edit_lesson_plans_form"):
            st.markdown("#### 생성된 차시별 계획 수정")

            lesson_plans = st.session_state.data.get('lesson_plans', [])
            edited_plans = []

            total_tabs = (total_hours + 9) // 10
            tabs = st.tabs([f"{i*10+1}~{min((i+1)*10, total_hours)}차시" for i in range(total_tabs)])

            for tab_idx, tab in enumerate(tabs):
                with tab:
                    start_idx = tab_idx * 10
                    end_idx = min(start_idx + 10, total_hours)

                    for i in range(start_idx, end_idx):
                        st.markdown(f"##### {i+1}차시")

                        col1, col2 = st.columns([1, 2])
                        with col1:
                            topic = st.text_input(
                                "학습주제",
                                value=lesson_plans[i].get('topic', ''),
                                key=f"topic_{i}",
                                help="이 차시의 주요 학습 주제를 입력하세요."
                            )
                            materials = st.text_input(
                                "교수학습자료",
                                value=lesson_plans[i].get('materials', ''),
                                key=f"materials_{i}",
                                help="필요한 교구와 자료를 입력하세요."
                            )

                        with col2:
                            content = st.text_area(
                                "학습내용",
                                value=lesson_plans[i].get('content', ''),
                                key=f"content_{i}",
                                height=100,
                                help="구체적인 학습 활동 내용을 입력하세요."
                            )

                        edited_plans.append({
                            "lesson_number": f"{i+1}",
                            "topic": topic,
                            "content": content,
                            "materials": materials
                        })
                        st.markdown("---")

            # 수정 및 다음 단계 버튼
            submit_button_edit = st.form_submit_button("수정사항 저장 및 다음 단계로", use_container_width=True)

        if submit_button_edit:
            with st.spinner("수정사항을 저장하고 다음 단계로 이동 중입니다..."):
                # 데이터 업데이트
                st.session_state.data['lesson_plans'] = edited_plans

                # 수정 완료 플래그 제거
                del st.session_state.generated_step_5

                st.success("차시별 계획이 저장되었습니다.")
                st.session_state.step = 6
                st.rerun()

    return False

###############################################################################
# 7. Excel 문서 생성 함수
###############################################################################
def create_excel_document():
    """
    현재 세션의 데이터를 기반으로 Excel 문서를 생성합니다.
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book

        # 셀 스타일 설정
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#E2E8F0',
            'border': 1,
            'text_wrap': True,
            'align': 'center',
            'valign': 'vcenter'
        })
        
        content_format = workbook.add_format({
            'text_wrap': True,
            'valign': 'top',
            'border': 1
        })

        # (1) 기본정보 시트
        basic_info = pd.DataFrame([{
            '학교급': st.session_state.data.get('school_type', ''),
            '대상학년': ', '.join(st.session_state.data.get('grades', [])),
            '총차시': st.session_state.data.get('total_hours', ''),
            '주당차시': st.session_state.data.get('weekly_hours', ''),
            '운영학기': ', '.join(st.session_state.data.get('semester', [])),
            '연계교과': ', '.join(st.session_state.data.get('subjects', [])),
            '활동명': st.session_state.data.get('activity_name', ''),
            '요구사항': st.session_state.data.get('requirements', ''),
            '필요성': st.session_state.data.get('necessity', ''),
            '개요': st.session_state.data.get('overview', ''),
            '성격': st.session_state.data.get('characteristics', '')
        }])
        basic_info.T.to_excel(writer, sheet_name='기본정보', header=['내용'])

        # (2) 목표/내용 시트
        goals_data = []
        for goal in st.session_state.data.get('goals', []):
            goals_data.append({'구분': '목표', '내용': goal})
        for idea in st.session_state.data.get('key_ideas', []):
            goals_data.append({'구분': '핵심아이디어', '내용': idea})
        pd.DataFrame(goals_data).to_excel(writer, sheet_name='목표및내용', index=False)

        # (3) 성취기준 시트
        standards_data = []
        for std in st.session_state.data.get('standards', []):
            for level in std['levels']:
                standards_data.append({
                    '성취기준': std['code'],
                    '설명': std['description'],
                    '수준': level['level'],
                    '수준별설명': level['description']
                })
        pd.DataFrame(standards_data).to_excel(writer, sheet_name='성취기준', index=False)

        # (4) 교수학습 및 평가 시트
        methods_data = []
        for method in st.session_state.data.get('teaching_methods', []):
            methods_data.append({
                '구분': '교수학습방법',
                '항목': method.get('method', ''),
                '설명': method.get('description', '')
            })
        for plan in st.session_state.data.get('assessment_plan', []):
            methods_data.append({
                '구분': '평가계획',
                '항목': plan.get('focus', ''),
                '설명': plan.get('description', '')
            })
        pd.DataFrame(methods_data).to_excel(writer, sheet_name='교수학습및평가', index=False)

        # (5) 차시별계획 시트
        lesson_plans_df = pd.DataFrame(st.session_state.data.get('lesson_plans', []))
        lesson_plans_df.columns = ['차시', '학습주제', '학습내용', '교수학습자료']  # 열 이름 한글화
        lesson_plans_df.to_excel(writer, sheet_name='차시별계획', index=False)

        # 모든 시트의 열 너비 조정
        for worksheet in writer.sheets.values():
            worksheet.set_column('A:A', 15)  # 첫 번째 열
            worksheet.set_column('B:B', 40)  # 두 번째 열
            worksheet.set_column('C:D', 20)  # 세 번째, 네 번째 열
            
            # 행 높이 자동 조정을 위한 설정
            worksheet.set_default_row(30)
            worksheet.set_row(0, 40)  # 헤더 행 높이

    return output.getvalue()

###############################################################################
# 8. 최종 검토 UI
###############################################################################
def show_final_review(vector_store):
    """최종 계획서 검토 UI"""
    st.title("최종 계획서 검토")
    
    try:
        data = st.session_state.data
        tabs = st.tabs(["기본정보", "목표/내용", "성취기준", "교수학습/평가", "차시별계획"])

        with tabs[0]:
            st.markdown("### 기본 정보")
            basic_info = {
                "학교급": data.get('school_type', ''),
                "대상 학년": ', '.join(data.get('grades', [])),
                "총 차시": f"{data.get('total_hours', '')}차시",
                "주당 차시": f"{data.get('weekly_hours', '')}차시",
                "운영 학기": ', '.join(data.get('semester', [])),
                "연계 교과": ', '.join(data.get('subjects', [])),
                "활동명": data.get('activity_name', ''),
                "요구사항": data.get('requirements', ''),
                "필요성": data.get('necessity', ''),
                "개요": data.get('overview', ''),
                "성격": data.get('characteristics', '')
            }
            for key, value in basic_info.items():
                st.markdown(f"**{key}**: {value}")

            st.button("기본정보 수정하기", key="edit_basic_info", on_click=lambda: set_step(1), use_container_width=True)

        with tabs[1]:
            st.markdown("### 목표 및 내용")
            st.markdown("#### 목표")
            for goal in data.get('goals', []):
                st.write(f"- {goal}")
            st.markdown("#### 활동 영역")
            st.write(data.get('domain', ''))
            st.markdown("#### 핵심 아이디어")
            for idea in data.get('key_ideas', []):
                st.write(f"- {idea}")

            st.button("목표/내용 수정하기", key="edit_goals_content", on_click=lambda: set_step(2), use_container_width=True)

        with tabs[2]:
            st.markdown("### 성취기준")
            for std in data.get('standards', []):
                st.markdown(f"**{std['code']}**: {std['description']}")
                st.markdown("##### 수준별 성취기준")
                for level in std['levels']:
                    st.write(f"- {level['level']} 수준: {level['description']}")
                st.markdown("---")

            st.button("성취기준 수정하기", key="edit_standards", on_click=lambda: set_step(3), use_container_width=True)

        with tabs[3]:
            st.markdown("### 교수학습 방법 및 평가계획")
            st.markdown("#### 교수학습 방법")
            for method in data.get('teaching_methods', []):
                st.write(f"- **{method['method']}**: {method['description']}")
            st.markdown("#### 평가계획")
            for assessment in data.get('assessment_plan', []):
                st.write(f"- **{assessment['focus']}**: {assessment['description']}")

            st.button("교수학습/평가계획 수정하기", key="edit_teaching_assessment", on_click=lambda: set_step(4), use_container_width=True)

        with tabs[4]:
            st.markdown("### 차시별 계획")
            lesson_plans_df = pd.DataFrame(data.get('lesson_plans', []))
            st.dataframe(
                lesson_plans_df,
                column_config={
                    "lesson_number": "차시",
                    "topic": "학습주제",
                    "content": "학습내용",
                    "materials": "교수학습자료"
                },
                hide_index=True,
                height=400
            )

            st.button("차시별 계획 수정하기", key="edit_lesson_plans", on_click=lambda: set_step(5), use_container_width=True)

        # 하단 버튼 그룹
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("모든 단계 수정하기", use_container_width=True):
                st.session_state.step = 1
                st.rerun()

        with col2:
            excel_data = create_excel_document()
            st.download_button(
                "📥 Excel 다운로드",
                excel_data,
                file_name=f"{data.get('activity_name', '학교자율시간계획서')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        with col3:
            if st.button("새로 만들기", use_container_width=True):
                st.session_state.clear()
                st.rerun()

    except Exception as e:
        st.error(f"최종 검토 화면 처리 중 오류가 발생했습니다: {str(e)}")

###############################################################################
# 9. 단계 이동 함수
###############################################################################
def set_step(step_number):
    """특정 단계로 이동하는 함수"""
    st.session_state.step = step_number
    # st.rerun()을 콜백 내에서 호출하지 않음. Streamlit이 자동으로 리런함.

###############################################################################
# 10. 메인 함수
###############################################################################
def main():
    """메인 함수: 애플리케이션의 전체 실행 흐름을 관리"""
    try:
        # 페이지 기본 설정
        set_page_config()

        # 세션 상태 초기화
        if 'data' not in st.session_state:
            st.session_state.data = {}
        if 'step' not in st.session_state:
            st.session_state.step = 1

        # 앱 제목
        st.title("2022 개정 교육과정 학교자율시간 계획서 생성기")
        
        # 진행 상황 표시
        show_progress()

        # 벡터 스토어 설정
        vector_store = setup_vector_store()
        if not vector_store:
            st.error("문서 임베딩에 실패했습니다. documents 폴더를 확인해주세요.")
            return

        # 현재 단계에 따른 UI 표시
        step_functions = {
            1: show_step_1,
            2: show_step_2,
            3: show_step_3,
            4: show_step_4,
            5: show_step_5,
            6: show_final_review
        }

        current_step = st.session_state.step
        step_function = step_functions.get(current_step)
        
        if step_function:
            step_function(vector_store)
        else:
            st.error("잘못된 단계입니다.")

    except Exception as e:
        st.error(f"애플리케이션 실행 중 오류가 발생했습니다: {str(e)}")
        if st.button("처음부터 다시 시작", use_container_width=True):
            st.session_state.clear()
            st.rerun()

# 애플리케이션 실행
if __name__ == "__main__":
    main()
