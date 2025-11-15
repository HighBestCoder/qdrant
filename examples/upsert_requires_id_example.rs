/// 这个示例证明Qdrant的upsert操作必须由客户端提供ID
/// 
/// 关键点：
/// 1. upsert_point() 方法签名要求传入 point_id: PointIdType
/// 2. 如果不提供ID，代码根本无法编译
/// 3. UpdateVectors API也要求每个PointVectors包含id字段

use segment::fixtures::segment::empty_segment;
use segment::segment::Segment;
use segment::types::{PointIdType, SeqNumberType};
use segment::data_types::vectors::only_default_vector;
use tempfile::tempdir;
use common::cpu::CpuBudget;

fn main() {
    // 创建测试segment
    let temp_dir = tempdir().unwrap();
    let mut segment = empty_segment(temp_dir.path());
    
    let hw_counter = CpuBudget::default();
    
    // ========== 示例1: Upsert必须提供ID ==========
    let vec1 = vec![1.0, 0.0, 1.0, 1.0];
    
    // 这是正确的调用方式：必须提供point_id
    segment
        .upsert_point(
            1 as SeqNumberType,          // operation number
            100.into(),                    // point_id (必须提供！)
            only_default_vector(&vec1),    // vectors
            &hw_counter,
        )
        .expect("Upsert should succeed");
    
    println!("✓ Point 100 upserted successfully with client-provided ID");
    
    // ========== 示例2: 尝试upsert相同ID会更新 ==========
    let vec2 = vec![2.0, 0.0, 2.0, 2.0];
    segment
        .upsert_point(
            2 as SeqNumberType,
            100.into(),                    // 使用相同的ID
            only_default_vector(&vec2),
            &hw_counter,
        )
        .expect("Upsert should update existing point");
    
    println!("✓ Point 100 updated successfully");
    
    // ========== 示例3: Update Vectors只能更新已存在的点 ==========
    let vec3 = vec![3.0, 0.0, 3.0, 3.0];
    
    // 更新已存在的点 - 成功
    segment
        .update_vectors(
            3 as SeqNumberType,
            100.into(),                    // 这个点已经存在
            only_default_vector(&vec3),
            &hw_counter,
        )
        .expect("Update existing point should succeed");
    
    println!("✓ Point 100 vectors updated successfully");
    
    // 尝试更新不存在的点 - 失败
    let result = segment.update_vectors(
        4 as SeqNumberType,
        999.into(),                        // 这个点不存在
        only_default_vector(&vec3),
        &hw_counter,
    );
    
    match result {
        Err(e) => println!("✓ Expected error when updating non-existent point: {:?}", e),
        Ok(_) => panic!("Should fail when updating non-existent point!"),
    }
    
    println!("\n总结:");
    println!("1. upsert_point() 必须由客户端提供 point_id");
    println!("2. 如果ID已存在，upsert会更新该点");
    println!("3. update_vectors() 只能更新已存在的点，否则返回错误");
    println!("4. Qdrant不会自动生成ID，所有ID都由客户端管理");
}
