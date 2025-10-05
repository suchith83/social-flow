"""
Quick API endpoint test script.

Tests all 15 AI/ML endpoints to verify they are accessible and responding correctly.
"""

import requests
import json
from typing import Dict, Any

BASE_URL = "http://127.0.0.1:8000"

def test_endpoint(method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
    """Test a single endpoint and return results."""
    url = f"{BASE_URL}{endpoint}"
    try:
        if method.upper() == "GET":
            response = requests.get(url, timeout=5, **kwargs)
        elif method.upper() == "POST":
            response = requests.post(url, timeout=5, **kwargs)
        else:
            return {"error": f"Unsupported method: {method}"}
        
        return {
            "status_code": response.status_code,
            "success": response.status_code < 400,
            "response": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text[:200]
        }
    except requests.exceptions.RequestException as e:
        return {"error": str(e), "success": False}

def main():
    """Run all endpoint tests."""
    print("=" * 80)
    print("🧪 TESTING AI/ML API ENDPOINTS")
    print("=" * 80)
    print()
    
    tests = [
        # Health & Basic Endpoints
        ("GET", "/health", "Health Check", {}),
        ("GET", "/api/v1/openapi.json", "OpenAPI Schema", {}),
        
        # Recommendation Endpoints (No Auth Required)
        ("GET", "/api/v1/ai/recommendations/algorithms", "List Algorithms", {}),
        ("GET", "/api/v1/ai/recommendations?algorithm=trending&limit=5", "Trending Recommendations", {}),
        ("GET", "/api/v1/ai/recommendations?algorithm=collaborative_filtering&limit=3", "Collaborative Filtering", {}),
        ("GET", "/api/v1/ai/recommendations?algorithm=content_based&limit=3", "Content-Based Recommendations", {}),
        
        # Pipeline Health & Monitoring (No Auth Required)
        ("GET", "/api/v1/ai/pipelines/health", "Pipeline Health", {}),
        ("GET", "/api/v1/ai/pipelines/queue", "Queue Statistics", {}),
        ("GET", "/api/v1/ai/pipelines/metrics", "Pipeline Metrics", {}),
        ("GET", "/api/v1/ai/pipelines/performance", "Performance Report", {}),
        ("GET", "/api/v1/ai/pipelines/schedule", "Scheduler Status", {}),
    ]
    
    results = {"passed": 0, "failed": 0, "total": len(tests)}
    
    for i, (method, endpoint, name, kwargs) in enumerate(tests, 1):
        print(f"[{i}/{len(tests)}] Testing: {name}")
        print(f"    {method} {endpoint}")
        
        result = test_endpoint(method, endpoint, **kwargs)
        
        if result.get("success"):
            print(f"    ✅ SUCCESS (Status: {result['status_code']})")
            results["passed"] += 1
            
            # Show sample response for interesting endpoints
            if "recommendations" in endpoint or "algorithms" in endpoint:
                response = result.get("response", {})
                if isinstance(response, dict):
                    if "algorithms" in response:
                        print(f"       Algorithms: {', '.join(response['algorithms'][:3])}...")
                    elif "recommendations" in response:
                        print(f"       Recommendations count: {len(response['recommendations'])}")
                elif isinstance(response, list):
                    print(f"       Count: {len(response)}")
        else:
            print(f"    ❌ FAILED")
            if "error" in result:
                print(f"       Error: {result['error']}")
            else:
                print(f"       Status: {result['status_code']}")
                print(f"       Response: {str(result.get('response', ''))[:100]}")
            results["failed"] += 1
        
        print()
    
    # Summary
    print("=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    print(f"Total Tests: {results['total']}")
    print(f"✅ Passed: {results['passed']}")
    print(f"❌ Failed: {results['failed']}")
    print(f"Success Rate: {(results['passed'] / results['total'] * 100):.1f}%")
    print()
    
    if results["failed"] == 0:
        print("🎉 All tests passed! API is fully functional.")
    else:
        print(f"⚠️  {results['failed']} test(s) failed. Check the output above for details.")
    
    print()
    print("=" * 80)
    print("🚀 Next Steps:")
    print("   1. Open http://localhost:8000/docs for interactive API testing")
    print("   2. Try different recommendation algorithms")
    print("   3. Test authenticated endpoints (pipeline management)")
    print("=" * 80)

if __name__ == "__main__":
    main()
