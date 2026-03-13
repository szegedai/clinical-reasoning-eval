import subprocess
import json

def check_usage():
    try:
        result = subprocess.run(
            ["bq", "query", "--format=json", "--nouse_legacy_sql", 
             "SELECT SUM(total_bytes_billed)/POW(1024, 4) as tb_billed FROM `region-us`.INFORMATION_SCHEMA.JOBS_BY_USER WHERE creation_time > TIMESTAMP_TRUNC(CURRENT_TIMESTAMP(), MONTH)"],
            capture_output=True, text=True, check=True
        )
        data = json.loads(result.stdout)
        if data and data[0]['tb_billed']:
            tb_billed = float(data[0]['tb_billed'])
        else:
            tb_billed = 0.0

        free_tb = 1.0
        remaining = max(0, free_tb - tb_billed)
        
        print(f"Data Billed This Month : {tb_billed*1024:.2f} GB ({tb_billed:.4f} TB)")
        print(f"Free Tier Remaining    : {remaining*1024:.2f} GB ({remaining:.4f} TB)")
        
    except Exception as e:
        print(f"Error checking BigQuery usage: {e}")

if __name__ == "__main__":
    check_usage()
