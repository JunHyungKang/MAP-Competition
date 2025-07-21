#!/usr/bin/env python3
"""
Dash EDA 대시보드 실행 스크립트
"""

import subprocess
import sys
import os

def run_dash_dashboard():
    """Dash 대시보드 실행"""
    try:
        print("🚀 Dash EDA 대시보드를 시작합니다...")
        print("📍 브라우저에서 http://localhost:8050 접속하세요")
        print("⏹️  종료하려면 Ctrl+C를 누르세요")
        print("-" * 50)
        
        # 현재 디렉토리를 eda 폴더로 변경
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
        # Dash 대시보드 실행
        subprocess.run([sys.executable, "eda_dashboard.py"])
        
    except KeyboardInterrupt:
        print("\n👋 대시보드를 종료합니다.")
    except Exception as e:
        print(f"❌ 오류가 발생했습니다: {e}")
        print("💡 해결 방법:")
        print("   1. pip install dash dash-bootstrap-components")
        print("   2. data/ 폴더에 CSV 파일이 있는지 확인")
        print("   3. 포트 8050이 사용 중인지 확인")

if __name__ == "__main__":
    run_dash_dashboard() 