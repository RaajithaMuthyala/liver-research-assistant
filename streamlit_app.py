# Or use any text editor to replace the contentimport streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

st.set_page_config(
    page_title="Liver Disease Research Assistant", 
    page_icon="🔬",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the fine-tuned model from Hugging Face"""
    try:
        # Try to load your fine-tuned model from Hugging Face
        model_name = "Raajitha/liver-research-model"
        
        st.info(f"🔄 Loading model from Hugging Face: {model_name}")
        
        # Load tokenizer and model from Hugging Face
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        st.success("✅ Fine-tuned model loaded from Hugging Face!")
        return model, tokenizer, True
        
    except Exception as e:
        # Fallback to base model if fine-tuned model fails
        st.warning(f"⚠️ Could not load fine-tuned model: {e}")
        st.info("🔄 Loading base model as fallback...")
        
        try:
            base_model_name = "microsoft/DialoGPT-medium"
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            st.warning("⚠️ Using base model (fine-tuned weights not found)")
            return model, tokenizer, True
            
        except Exception as base_error:
            st.error(f"❌ Failed to load any model: {base_error}")
            return None, None, False

def generate_answer(question, model, tokenizer):
    """Generate answer using the model"""
    
    prompt = f"### Instruction:\nAnswer the research question about liver disease and social determinants of health based on scientific evidence.\n\n### Input:\n{question}\n\n### Response:\n"
    
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=inputs.shape[1] + 150,  # Increased for better responses
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
    
    # Model info in sidebar
    st.sidebar.header("🤖 Model Status")
    if model_loaded:
        st.sidebar.success("🟢 Model: Ready")
        st.sidebar.info("📍 Source: Hugging Face Hub")
    else:
        st.sidebar.error("🔴 Model: Error")
    
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
                    try:
                        answer = generate_answer(question, model, tokenizer)
                        st.success("✅ **Answer:**")
                        st.write(answer)
                    except Exception as e:
                        st.error(f"Error generating answer: {e}")
                
            elif not model_loaded:
                st.error("Model not loaded properly")
            else:
                st.warning("Please enter a question")
    
    with col2:
        st.header("📊 System Info")
        
        st.info("""
        **📚 Trained on:**
        • 2,608 research abstracts
        • Liver disease literature  
        • Social determinants of health
        
        **💡 Example questions:**
        • How does income affect liver health?
        • What social factors influence HCC?
        • How does education impact liver disease?
        • What are the risk factors for NAFLD?
        """)
        
        # Show model details
        if model_loaded:
            with st.expander("🔧 Technical Details"):
                st.text("Model: Fine-tuned DialoGPT")
                st.text("Source: Hugging Face Hub")
                st.text("Framework: PyTorch + Transformers")
    
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
