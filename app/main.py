from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, Optional, List, Tuple
import uvicorn
import logging
from datetime import datetime
from .core.data_manager import DataManager
from .core.enhanced_analyzer import EnhancedAnalyzer, AnalysisResult
from .utils.monitoring import ResourceMonitor
from config.settings import (
    MINIMUM_SCORES,
    MEMORY_THRESHOLDS,
    ERROR_THRESHOLDS,
    PERFORMANCE_THRESHOLDS
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Enhanced ATS Resume Optimizer",
    description="Advanced resume optimization system with industry-specific ATS matching and keyword optimization"
)

# Initialize components
data_manager = DataManager()
analyzer = EnhancedAnalyzer(data_manager)
monitor = ResourceMonitor()

class OptimizationRequest(BaseModel):
    resume_text: str = Field(..., description="The full text of the resume")
    job_description: str = Field(..., description="The full text of the job description")
    target_job: Optional[str] = Field(None, description="Specific job title to target")
    target_industry: Optional[str] = Field(None, description="Specific industry to target")
    optimize_keywords: bool = Field(True, description="Whether to optimize keywords")
    restructure_content: bool = Field(True, description="Whether to restructure content")

class OptimizationResponse(BaseModel):
    optimized_resume: str
    ats_score: float
    industry_match_score: float
    experience_match_score: float
    keyword_density: float
    skill_matches: Dict[str, float]
    missing_critical_skills: List[str]
    improvement_suggestions: List[str]
    salary_match: bool
    location_match: bool
    overall_score: float
    performance_metrics: Dict[str, float]

class HealthCheckResponse(BaseModel):
    status: str
    memory_usage: float
    error_rate: float
    last_execution_time: float
    components_status: Dict[str, bool]
    timestamp: datetime

async def get_performance_metrics():
    """Dependency to get current performance metrics"""
    return {
        "memory_usage": monitor.process.memory_percent(),
        "cpu_usage": monitor.process.cpu_percent(),
        "error_rate": monitor._calculate_error_rate()
    }

@app.post("/optimize", response_model=OptimizationResponse)
async def optimize_resume(
    request: OptimizationRequest,
    performance_metrics: Dict[str, float] = Depends(get_performance_metrics)
):
    try:
        with monitor.track_performance():
            # Analyze resume
            result = analyzer.analyze_resume(
                resume_text=request.resume_text,
                job_description=request.job_description,
                target_job=request.target_job,
                target_industry=request.target_industry
            )
            
            # Check if results meet minimum requirements
            if result.ats_score < MINIMUM_SCORES["ATS_SCORE"]:
                logger.warning(f"Low ATS score: {result.ats_score:.2f}")
            
            if result.industry_match_score < MINIMUM_SCORES["INDUSTRY_MATCH"]:
                logger.warning(f"Low industry match: {result.industry_match_score:.2f}")
            
            return OptimizationResponse(
                **result.__dict__,
                performance_metrics=performance_metrics
            )
            
    except Exception as e:
        logger.error(f"Optimization failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "metrics": performance_metrics
            }
        )

@app.get("/health", response_model=HealthCheckResponse)
async def health_check(
    performance_metrics: Dict[str, float] = Depends(get_performance_metrics)
):
    components_status = {
        "data_manager": data_manager is not None,
        "analyzer": analyzer is not None,
        "monitor": monitor is not None
    }
    
    status = "healthy"
    if (performance_metrics["memory_usage"] > MEMORY_THRESHOLDS["WARNING"] or
        performance_metrics["error_rate"] > ERROR_THRESHOLDS["WARNING"]):
        status = "degraded"
        
    if (performance_metrics["memory_usage"] > MEMORY_THRESHOLDS["CRITICAL"] or
        performance_metrics["error_rate"] > ERROR_THRESHOLDS["CRITICAL"]):
        status = "critical"
    
    return HealthCheckResponse(
        status=status,
        memory_usage=performance_metrics["memory_usage"],
        error_rate=performance_metrics["error_rate"],
        last_execution_time=monitor.last_execution_time,
        components_status=components_status,
        timestamp=datetime.now()
    )

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=4  # Optimized for M4 Pro 12-core CPU
    )
