import streamlit as st
import subprocess
import os
import pandas as pd

st.set_page_config(page_title="Redrob AI Ranker", layout="wide")
st.title("Redrob AI Ranking Engine - Sandbox")

st.markdown("""
Upload a subset of `candidates.jsonl` to test the engine. 
The engine will apply our O(N log K) Min-Heap architecture, process the redrob_signals, and generate the top 100 CSV in seconds.
""")

uploaded_file = st.file_uploader("Upload test_candidates.jsonl", type=["jsonl", "json"])

if uploaded_file is not None:
    if st.button("▶️ Run Ranking Engine"):
        with st.spinner("Executing rank.py..."):
            temp_input = "temp_sandbox_candidates.jsonl"
            temp_output = "sandbox_submission.csv"
            
            # Save uploaded file
            with open(temp_input, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            # Execute the ranking engine
            cmd = ["python", "rank.py", "--candidates", temp_input, "--output", temp_output]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            st.subheader("Console Output")
            st.code(result.stdout)
            
            if os.path.exists(temp_output):
                st.success("✅ Ranking Complete!")
                df = pd.read_csv(temp_output)
                st.dataframe(df)
                
                with open(temp_output, "rb") as f:
                    st.download_button("⬇️ Download submission.csv", f, file_name="submission.csv")
            else:
                st.error("Engine failed to produce output. See traceback below.")
                st.code(result.stderr)
