import json
import os

def format_leetcode_sample(problem_statement, solution_code):
    messages = [
        {
            "role": "system", 
            "content": "You are an expert Python algorithmic coding assistant. Write clean, optimal code with brief explanations."
        },
        {
            "role": "user", 
            "content": f"Solve this LeetCode problem:\n\n{problem_statement}"
        },
        {
            "role": "assistant", 
            "content": f"```python\n{solution_code}\n```"
        }
    ]
    return {"messages": messages}

if __name__ == "__main__":
    sample_problem = "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target."
    sample_solution = (
        "def twoSum(nums, target):\n"
        "    seen = {}\n"
        "    for i, num in enumerate(nums):\n"
        "        diff = target - num\n"
        "        if diff in seen:\n"
        "            return [seen[diff], i]\n"
        "        seen[num] = i"
    )
    
    formatted = format_leetcode_sample(sample_problem, sample_solution)
    print(json.dumps(formatted, indent=2))
