#!/usr/bin/env python3
"""
Streamlit EDA 대시보드 실행 스크립트
"""

import subprocess
import sys
import os

def run_streamlit_dashboard():
    """Streamlit 대시보드 실행"""
    try:
        print("🚀 Streamlit EDA 대시보드를 시작합니다...")
        print("📍 브라우저에서 http://localhost:8501 접속하세요")
        print("⏹️  종료하려면 Ctrl+C를 누르세요")
        print("-" * 50)
        
        # 현재 디렉토리를 eda 폴더로 변경
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
        # Streamlit 대시보드 실행
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_eda.py", 
            "--server.port", "8501"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 대시보드를 종료합니다.")
    except Exception as e:
        print(f"❌ 오류가 발생했습니다: {e}")
        print("💡 해결 방법:")
        print("   1. pip install streamlit")
        print("   2. data/ 폴더에 CSV 파일이 있는지 확인")
        print("   3. 포트 8501이 사용 중인지 확인")

if __name__ == "__main__":
    run_streamlit_dashboard() 