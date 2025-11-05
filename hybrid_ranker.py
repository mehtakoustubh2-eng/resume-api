import os
import requests
import numpy as np
from typing import List, Dict, Tuple, Optional
import google.generativeai as genai
import re

class HybridResumeRanker:
    def __init__(self):
        # Use feature-extraction pipeline
        self.hf_url = "https://router.huggingface.co/hf-inference/models/BAAI/bge-small-en"
        self.hf_headers = {"Authorization": f"Bearer {os.getenv('HF_API_TOKEN')}","Content-Type": "application/json"}
        
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.gemini = genai.GenerativeModel('models/gemini-2.0-flash')
    
    def process(self, job_desc: str, resumes: List[str], top_k: int = 10, 
               sbert_filter_size: Optional[int] = None) -> Dict:
        """
        ALWAYS use hybrid SBERT + Gemini approach for consistent results
        
        Args:
            job_desc: Job description text
            resumes: List of resume texts
            top_k: Final number of top resumes to return
            sbert_filter_size: Number of resumes for SBERT to pass to Gemini
        """
        print(" HYBRID MODE: Using SBERT + Gemini for consistent scoring")
        
        # Determine SBERT filter size
        filter_size = sbert_filter_size if sbert_filter_size is not None else top_k
        
        print(f" Processing {len(resumes)} resumes")
        print(f" SBERT filter: {len(resumes)} → {filter_size} candidates")
        print(f" Final return: top {top_k} ranked candidates")
        
        # Step 1: SBERT filtering via Hugging Face API
        top_resumes = self._sbert_filter(job_desc, resumes, filter_size)
        
        # Step 2: Gemini scoring
        ranked_results = self._get_gemini_scores(job_desc, top_resumes)
        
        # Final top-k selection
        final_rankings = ranked_results[:top_k]
        
        print(f" Successfully processed {len(resumes)} resumes → {len(final_rankings)} final rankings")
        
        return {
            "job_description": job_desc,
            "total_processed": len(resumes),
            "sbert_filtered": len(top_resumes),
            "final_top_k": top_k,
            "rankings": final_rankings
        }
    
    def _sbert_filter(self, job_desc: str, resumes: List[str], top_k: int) -> List[Dict]:
        """SBERT filtering via Hugging Face API"""
        print(f" Getting job description embedding...")
        job_emb = self.get_embedding(job_desc)
        if not job_emb:
            print(" Failed to get job embedding")
            return []
        
        print(f" Calculating similarities for {len(resumes)} resumes...")
        similarities = []
        for i, resume in enumerate(resumes):
            resume_emb = self.get_embedding(resume)
            if resume_emb:
                similarity = np.dot(job_emb, resume_emb) / (np.linalg.norm(job_emb) * np.linalg.norm(resume_emb))
                similarities.append((i, resume, similarity))
            else:
                print(f"  Failed to get embedding for resume {i}")
        
        # Get top-k candidates
        similarities.sort(key=lambda x: x[2], reverse=True)
        top_candidates = [{
            'index': idx,
            'resume_text': resume,
            'sbert_similarity': round(score * 100, 1)  # Convert to 0-10 scale
        } for idx, resume, score in similarities[:top_k]]
        
        print(f" SBERT found {len(top_candidates)} candidates with similarities: {[c['sbert_similarity'] for c in top_candidates]}")
        return top_candidates
    
    def _get_gemini_scores(self, job_desc: str, top_resumes: List[Dict]) -> List[Dict]:
        """Gemini scores only the SBERT-filtered resumes"""
        print(f" Gemini analyzing {len(top_resumes)} resumes...")
        results = []
        
        for i, resume_data in enumerate(top_resumes):
            resume_text = resume_data['resume_text']
            original_index = resume_data['index']
            sbert_score = resume_data['sbert_similarity']
        
            print(f"    Processing resume {i+1}/{len(top_resumes)} (index: {original_index})")
            
            prompt = f"""
            ACT as an expert resume screening AI. Evaluate how well this resume matches the job description.
            
            JOB DESCRIPTION:
            {job_desc}
            
            RESUME:
            {resume_text}
            
            Provide your evaluation in this EXACT format:
            Score: [number between 1-10]
            Explanation: [2-3 sentence explanation of strengths and weaknesses]
            
            Be strict but fair in your evaluation.
            """
            
            try:
                response = self.gemini.generate_content(prompt)
                
                #  ADD DEBUG LOGGING
                print(f"      Raw Gemini response: {response.text}")
                
                gemini_score, explanation = self._parse_gemini_response(response.text)
                
                print(f"      Scored: {gemini_score}/10")
                print(f"      Explanation: {explanation}")
                
                results.append({
                    'resume_index': original_index,
                    'gemini_score': gemini_score,
                    'sbert_similarity': sbert_score,
                    'explanation': explanation,
                    'resume_preview': resume_text[:200] + "..." if len(resume_text) > 200 else resume_text
                })
                
            except Exception as e:
                print(f"     Error: {e}")
                results.append({
                    'resume_index': original_index,
                    'gemini_score': 5.0,
                    'sbert_similarity': sbert_score,
                    'explanation': "Unable to evaluate this resume",
                    'resume_preview': resume_text[:200] + "..." if len(resume_text) > 200 else resume_text
                })
        
        results.sort(key=lambda x: x['gemini_score'], reverse=True)
        print(f" Gemini ranking complete. Top score: {results[0]['gemini_score'] if results else 'N/A'}")
        return results
    
    def _parse_gemini_response(self, response_text: str) -> Tuple[float, str]:
        """Parse Gemini response to extract score and explanation"""
        try:
            print(f" PARSING GEMINI RESPONSE: {response_text}")
            
            lines = response_text.strip().split('\n')
            score_line = next((line for line in lines if line.lower().startswith('score:')), None)
            explanation_line = next((line for line in lines if line.lower().startswith('explanation:')), None)
            
            score = 5.0  # Default score
            
            if score_line:
                #  IMPROVED PARSING LOGIC
                score_text = score_line.split(':')[1].strip()
                
                # Extract first number from the score text
                numbers = re.findall(r'\d+\.?\d*', score_text)
                
                if numbers:
                    score = float(numbers[0])
                    # Ensure score is between 1-10
                    score = max(1, min(10, score))
                    print(f" EXTRACTED SCORE: {score} from '{score_text}'")
                else:
                    print(f" NO NUMBER FOUND in score line: '{score_line}'")
            else:
                print(f" NO SCORE LINE FOUND in response")
                
            if explanation_line:
                explanation = explanation_line.split(':', 1)[1].strip()  # Use max 1 split
            else:
                explanation = "No explanation provided"
                print(f" NO EXPLANATION LINE FOUND in response")
                
            return score, explanation
            
        except Exception as e:
            print(f" Error parsing Gemini response: {e}")
            return 5.0, "Error in evaluation"
    
    def get_embedding(self, text: str):
        """Get embeddings from Hugging Face API"""
        try:
            response = requests.post(
                self.hf_url, 
                headers=self.hf_headers, 
                json={"inputs": text},
                timeout=30
            )
            if response.status_code == 200:
                return response.json()
            else:
                print(f" HF API Error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f" HF API Exception: {e}")
            return None