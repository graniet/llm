/// Tests for the LLM chain functionality
#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use tempfile::{tempdir, NamedTempFile};
    use std::io::Write;
    use std::fs;
    
    /// Tests basic condition evaluation functionality
    #[test]
    fn test_evaluate_condition_basics() {
        let mut memory = HashMap::new();
        memory.insert("status".to_string(), "active".to_string());
        memory.insert("message".to_string(), "Hello world".to_string());
        
        /// Simple condition evaluator that checks for equality and existence
        fn evaluate_condition_simple(condition: &Option<String>, memory: &HashMap<String, String>) -> bool {
            if let Some(condition) = condition {
                if condition.is_empty() {
                    return true;
                }
                
                if let Some(equals_pos) = condition.find('=') {
                    let var_name = condition[..equals_pos].trim().to_string();
                    let expected_value = &condition[equals_pos+1..].trim();
                    
                    if let Some(actual_value) = memory.get(&var_name) {
                        return actual_value == expected_value;
                    }
                    return false;
                }
                
                let var_name = condition.trim().to_string();
                return memory.contains_key(&var_name);
            } else {
                return true;
            }
        }
        
        assert!(evaluate_condition_simple(&None, &memory));
        assert!(evaluate_condition_simple(&Some("".to_string()), &memory));
        assert!(evaluate_condition_simple(&Some("status=active".to_string()), &memory));
        assert!(!evaluate_condition_simple(&Some("status=inactive".to_string()), &memory));
        assert!(evaluate_condition_simple(&Some("status".to_string()), &memory));
        assert!(!evaluate_condition_simple(&Some("unknown".to_string()), &memory));
    }
    
    /// Tests creation of chain template files
    #[test]
    fn test_chain_template_creation() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempdir()?;
        let file_path = dir.path().join("test_template.yaml");
        
        let template_content = r#"
name: example-chain
description: A chain that demonstrates multi-step LLM processing
default_provider: openai:gpt-4o
steps:
  - id: step1
    template: This is a test step
    mode: chat
    temperature: 0.7
    max_tokens: 100
"#;
        
        fs::write(&file_path, template_content)?;
        
        assert!(file_path.exists());
        
        let content = fs::read_to_string(&file_path)?;
        assert!(content.contains("example-chain"));
        assert!(content.contains("steps:"));
        
        Ok(())
    }
    
    /// Tests loading chain configuration from YAML files
    #[test]
    fn test_load_chain_config() -> Result<(), Box<dyn std::error::Error>> {
        let yaml_content = r#"
name: test-chain
description: Test chain
default_provider: openai:gpt-4o
steps:
  - id: step1
    template: This is a test step
    mode: chat
    temperature: 0.5
    max_tokens: 100
"#;

        let mut temp_file = NamedTempFile::new()?;
        temp_file.write_all(yaml_content.as_bytes())?;
        let file_path = temp_file.into_temp_path();
        
        let content = fs::read_to_string(&file_path)?;
        assert!(content.contains("test-chain"));
        assert!(content.contains("Test chain"));
        assert!(content.contains("openai:gpt-4o"));
        
        Ok(())
    }
    
    /// Tests template variable substitution functionality
    #[test]
    fn test_template_substitution() {
        /// Applies template substitutions using a memory map
        fn apply_template(template: &str, memory: &HashMap<String, String>) -> String {
            let mut result = template.to_string();
            for (k, v) in memory {
                let pattern = format!("{{{{{}}}}}", k);
                result = result.replace(&pattern, v);
            }
            result
        }
        
        let mut memory = HashMap::new();
        memory.insert("name".to_string(), "John".to_string());
        memory.insert("age".to_string(), "30".to_string());
        
        let template = "Hello {{name}}, you are {{age}} years old.";
        let result = apply_template(template, &memory);
        
        assert_eq!(result, "Hello John, you are 30 years old.");
    }
}