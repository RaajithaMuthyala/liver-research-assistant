import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import re

st.set_page_config(
    page_title="Liver Disease Research Assistant", 
    page_icon="🔬",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the fine-tuned model"""
    try:
        # Load your fine-tuned model (we'll update this path later)
        model_path = "./liver-model-final"
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        base_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/DialoGPT-medium",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Try to load fine-tuned weights if available
        try:
            model = PeftModel.from_pretrained(base_model, model_path)
            st.success("✅ Fine-tuned model loaded successfully!")
        except:
            model = base_model
            st.warning("⚠️ Using base model (fine-tuned weights not found)")
        
        return model, tokenizer, True
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, False

def generate_answer(question, model, tokenizer):
    """Generate answer using the model"""
    
    prompt = f"### Instruction:\nAnswer the research question about liver disease and social determinants of health based on scientific evidence.\n\n### Input:\n{question}\n\n### Response:\n"
    
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=inputs.shape[1] + 100,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part
    if "### Response:" in response:
        answer = response.split("### Response:")[-1].strip()
    else:
        answer = response
    
    return answer

def analyze_csv(df):
    """Analyze uploaded CSV file"""
    analysis = {}
    
    for col in df.columns:
        sample_data = df[col].dropna().head(5).astype(str)
        
        analysis[col] = {
            'type': str(df[col].dtype),
            'unique_count': df[col].nunique(),
            'null_count': df[col].isnull().sum(),
            'samples': sample_data.tolist()
        }
    
    return analysis

def main():
    st.title("🔬 Liver Disease & SDOH Research Assistant")
    st.markdown("*AI-powered Q&A system trained on liver disease research*")
    
    # Load model
    model, tokenizer, model_loaded = load_model()
    
    # Sidebar for file upload
    st.sidebar.header("📁 Upload Research Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file for analysis",
        type=['csv']
    )
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("❓ Ask Your Research Question")
        
        question = st.text_area(
            "Enter your question:",
            placeholder="Example: How do socioeconomic factors affect liver disease outcomes?",
            height=100
        )
        
        if st.button("🔍 Get Answer", type="primary"):
            if question and model_loaded:
                with st.spinner("Generating answer..."):
                    answer = generate_answer(question, model, tokenizer)
                
                st.success("✅ **Answer:**")
                st.write(answer)
                
            elif not model_loaded:
                st.error("Model not loaded properly")
            else:
                st.warning("Please enter a question")
    
    with col2:
        st.header("📊 System Info")
        
        if model_loaded:
            st.success("🟢 Model: Ready")
        else:
            st.error("🔴 Model: Error")
        
        st.info("""
        **📚 Trained on:**
        • 2,608 research abstracts
        • Liver disease literature  
        • Social determinants of health
        
        **💡 Example questions:**
        • How does income affect liver health?
        • What social factors influence HCC?
        • How does education impact liver disease?
        """)
    
    # CSV Analysis section
    if uploaded_file is not None:
        st.header("📈 CSV File Analysis")
        
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
            
            # Show analysis
            analysis = analyze_csv(df)
            
            for col, info in analysis.items():
                with st.expander(f"📋 Column: {col}"):
                    st.write(f"**Type:** {info['type']}")
                    st.write(f"**Unique values:** {info['unique_count']}")
                    st.write(f"**Missing values:** {info['null_count']}")
                    st.write("**Sample values:**")
                    for sample in info['samples']:
                        st.text(f"• {str(sample)[:100]}...")
            
            # Show data preview
            st.subheader("👀 Data Preview")
            st.dataframe(df.head())
            
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

if __name__ == "__main__":
    main()
