use serde_json::Value;

use crate::error::LLMError;

const DESCRIPTION: &str = "description";
const UNSUPPORTED_BOUNDS: &[&str] = &["maxLength", "maxItems"];
const SCHEMA_MAPS: &[&str] = &[
    "properties",
    "patternProperties",
    "$defs",
    "definitions",
    "dependentSchemas",
];
const SCHEMA_ARRAYS: &[&str] = &["allOf", "anyOf", "oneOf", "prefixItems"];
const CHILD_SCHEMAS: &[&str] = &[
    "items",
    "additionalItems",
    "additionalProperties",
    "contains",
    "not",
    "if",
    "then",
    "else",
    "propertyNames",
    "unevaluatedItems",
    "unevaluatedProperties",
];

/// Adapt schema positions only: property names, enum values, and examples are data.
/// The original schema must still be enforced locally after generation.
pub(super) fn adapt_schema(schema: &Value) -> Value {
    let Value::Object(fields) = schema else {
        return schema.clone();
    };
    let mut adapted = fields.clone();
    let constraints = UNSUPPORTED_BOUNDS
        .iter()
        .filter_map(|key| adapted.remove(*key).map(|bound| format!("{key} = {bound}")))
        .collect::<Vec<_>>();
    if !constraints.is_empty() {
        let description = fields
            .get(DESCRIPTION)
            .and_then(Value::as_str)
            .unwrap_or_default();
        adapted.insert(
            DESCRIPTION.into(),
            Value::String(
                format!(
                    "{description} Required constraints: {}.",
                    constraints.join(", ")
                )
                .trim()
                .into(),
            ),
        );
    }
    for key in SCHEMA_MAPS {
        if let Some(Value::Object(children)) = fields.get(*key) {
            adapted.insert(
                (*key).into(),
                Value::Object(
                    children
                        .iter()
                        .map(|(name, child)| (name.clone(), adapt_schema(child)))
                        .collect(),
                ),
            );
        }
    }
    for key in SCHEMA_ARRAYS {
        if let Some(Value::Array(children)) = fields.get(*key) {
            adapted.insert(
                (*key).into(),
                Value::Array(children.iter().map(adapt_schema).collect()),
            );
        }
    }
    for key in CHILD_SCHEMAS {
        if let Some(child) = fields.get(*key) {
            let child = match child {
                Value::Array(children) => Value::Array(children.iter().map(adapt_schema).collect()),
                _ => adapt_schema(child),
            };
            adapted.insert((*key).into(), child);
        }
    }
    // Anthropic's native structured output rejects any object schema whose
    // additionalProperties isn't explicitly false (400: "'additionalProperties'
    // must be explicitly set to false"), regardless of what the caller's
    // original schema declared. Force it after the recursive walk above so it
    // overrides whatever CHILD_SCHEMAS just adapted additionalProperties to.
    let is_object = fields.get("type").and_then(Value::as_str) == Some("object")
        || fields.contains_key("properties");
    if is_object {
        adapted.insert("additionalProperties".into(), Value::Bool(false));
    }
    Value::Object(adapted)
}

/// Detects a JSON Schema construct that Anthropic's native `output_config`
/// cannot express, so the caller can fall back to a prompt-based JSON
/// instruction instead of either a 400 from the API or a schema silently
/// narrowed by `adapt_schema` (forcing `additionalProperties: false` onto a
/// caller-intended free-form object would make it accept only `{}`).
///
/// Returns a human-readable description of the first offending construct
/// found, walking the same schema positions as `adapt_schema`.
pub(super) fn unsupported_native_construct(schema: &Value) -> Option<String> {
    match schema {
        Value::Bool(_) => Some("a boolean sub-schema (`true`/`false`)".to_string()),
        Value::Object(fields) => {
            if fields.is_empty() {
                return Some("an empty schema ({}) that accepts any JSON value".to_string());
            }
            if let Some(additional) = fields.get("additionalProperties") {
                if additional != &Value::Bool(false) {
                    return Some(
                        "'additionalProperties' set to something other than false".to_string(),
                    );
                }
            }
            if fields.contains_key("patternProperties") {
                return Some("'patternProperties'".to_string());
            }
            // additionalProperties defaults to `true` (open) per JSON Schema
            // when the keyword is absent, exactly like an explicit `true`
            // above - so an object with no *named* properties (missing
            // `properties`, or `properties: {}`) is only genuinely closed,
            // and safe to hand to adapt_schema, when it explicitly sets
            // `additionalProperties: false` itself.
            let is_object = fields.get("type").and_then(Value::as_str) == Some("object")
                || fields.contains_key("properties");
            if is_object {
                let has_named_properties = fields
                    .get("properties")
                    .and_then(Value::as_object)
                    .is_some_and(|properties| !properties.is_empty());
                let explicitly_closed =
                    fields.get("additionalProperties") == Some(&Value::Bool(false));
                if !has_named_properties && !explicitly_closed {
                    return Some(
                        "an object schema with no named properties that isn't explicitly \
                         closed with additionalProperties: false (free-form object)"
                            .to_string(),
                    );
                }
            }
            for key in SCHEMA_MAPS {
                if let Some(Value::Object(children)) = fields.get(*key) {
                    if let Some(reason) = children.values().find_map(unsupported_native_construct) {
                        return Some(reason);
                    }
                }
            }
            for key in SCHEMA_ARRAYS {
                if let Some(Value::Array(children)) = fields.get(*key) {
                    if let Some(reason) = children.iter().find_map(unsupported_native_construct) {
                        return Some(reason);
                    }
                }
            }
            for key in CHILD_SCHEMAS {
                // additionalProperties is already checked above; `false` is a
                // valid boolean value there, not an unsupported construct.
                if *key == "additionalProperties" {
                    continue;
                }
                if let Some(child) = fields.get(*key) {
                    let reason = match child {
                        Value::Array(children) => {
                            children.iter().find_map(unsupported_native_construct)
                        }
                        _ => unsupported_native_construct(child),
                    };
                    if reason.is_some() {
                        return reason;
                    }
                }
            }
            None
        }
        _ => None,
    }
}

/// Renders a schema as a system-prompt instruction for the prompt-based
/// fallback: models asked to emit JSON matching a schema they were never
/// shown will otherwise guess the shape.
pub(super) fn schema_prompt_instruction(schema: &Value) -> String {
    format!(
        "Respond with a single JSON value only - no markdown code fences, no prose before or \
         after it - that satisfies exactly this JSON Schema:\n{}",
        serde_json::to_string_pretty(schema).unwrap_or_else(|_| schema.to_string())
    )
}

pub(super) fn validate_output(
    validator: &jsonschema::JSONSchema,
    text: &str,
) -> Result<(), LLMError> {
    let value: Value = serde_json::from_str(text).map_err(|error| {
        LLMError::ProviderError(format!(
            "Anthropic structured output is not valid JSON: {error}"
        ))
    })?;
    if let Err(mut errors) = validator.validate(&value) {
        if let Some(error) = errors.next() {
            // Report paths, not the potentially sensitive generated value.
            return Err(LLMError::ProviderError(format!(
                "Anthropic structured output violates the original schema at {} (schema {})",
                error.instance_path, error.schema_path
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn adapts_draft_bounds_without_losing_structure_or_original_validation() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string", "maxLength": 3, "description": "Draft name"},
                "steps": {"type": "array", "minItems": 1, "maxItems": 1, "items": {
                    "type": "object", "properties": {"name": {"type": "string", "maxLength": 3}},
                    "required": ["name"], "additionalProperties": false
                }}
            }, "required": ["name", "steps"], "additionalProperties": false
        });
        let adapted = adapt_schema(&schema);
        assert!(adapted["properties"]["name"].get("maxLength").is_none());
        assert!(adapted["properties"]["steps"].get("maxItems").is_none());
        assert!(
            adapted["properties"]["steps"]["items"]["properties"]["name"]
                .get("maxLength")
                .is_none()
        );
        assert_eq!(adapted["properties"]["steps"]["minItems"], 1);
        assert_eq!(adapted["required"], schema["required"]);
        assert_eq!(adapted["additionalProperties"], false);
        assert_eq!(
            adapted["properties"]["name"]["description"],
            "Draft name Required constraints: maxLength = 3."
        );
        let validator = jsonschema::JSONSchema::compile(&schema).unwrap();
        assert!(
            validate_output(&validator, r#"{"name":"猫猫猫","steps":[{"name":"ok"}]}"#).is_ok()
        );
        for invalid in [
            r#"{"name":"long","steps":[{"name":"ok"}]}"#,
            r#"{"name":"ok","steps":[{"name":"ok"},{"name":"ok"}]}"#,
            r#"{"name":"ok","steps":[{"name":"long"}]}"#,
            r#"{"name":"ok"}"#,
            "not JSON",
        ] {
            assert!(validate_output(&validator, invalid).is_err(), "{invalid}");
        }
        assert_eq!(schema["properties"]["name"]["maxLength"], 3);
    }

    #[test]
    fn follows_definitions_and_unions_but_preserves_property_names_and_literal_data() {
        let schema = json!({
            "$defs": {"label": {"type": "string", "maxLength": 3}},
            "type": "object", "properties": {
                "maxItems": {"anyOf": [{"$ref": "#/$defs/label"}, {"type": "null"}]},
                "maxLength": {"type": "array", "maxItems": 1, "items": {"type": "string"}}
            },
            "examples": [{"maxLength": 17}], "const": {"maxItems": 9}
        });
        let adapted = adapt_schema(&schema);
        assert!(adapted["$defs"]["label"].get("maxLength").is_none());
        assert_eq!(
            adapted["properties"]["maxItems"],
            schema["properties"]["maxItems"]
        );
        assert!(adapted["properties"]["maxLength"].get("maxItems").is_none());
        assert_eq!(adapted["examples"], schema["examples"]);
        assert_eq!(adapted["const"], schema["const"]);
    }

    #[test]
    fn forces_additional_properties_false_on_every_object_even_when_absent_or_true() {
        let schema = json!({
            "type": "object",
            "properties": {
                "user": {
                    "properties": {"id": {"type": "string"}}
                },
                "meta": {"type": "object", "additionalProperties": true}
            }
        });
        let adapted = adapt_schema(&schema);
        assert_eq!(adapted["additionalProperties"], false);
        assert_eq!(adapted["properties"]["user"]["additionalProperties"], false);
        assert_eq!(adapted["properties"]["meta"]["additionalProperties"], false);
        // Non-object schemas are untouched.
        let string_schema = json!({"type": "string"});
        assert!(adapt_schema(&string_schema)
            .get("additionalProperties")
            .is_none());
    }

    #[test]
    fn flags_constructs_anthropic_cannot_express() {
        for schema in [
            json!({}),
            json!(true),
            json!({"type": "object"}),
            json!({"type": "object", "properties": {}, "additionalProperties": true}),
            json!({"type": "object", "properties": {}, "patternProperties": {"^x-": {"type": "string"}}}),
        ] {
            assert!(
                unsupported_native_construct(&schema).is_some(),
                "expected {schema} to be flagged"
            );
        }
    }

    #[test]
    fn flags_the_real_planner_schema_shape() {
        // app-workflow-manager's base_property_schema declares `value` as an
        // open `{}` and several fields as free-form objects - see
        // inxm-ai/app-tgi-core#72.
        let schema = json!({
            "type": "object",
            "properties": {
                "plan_properties": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"value": {}},
                        "required": ["value"],
                        "additionalProperties": false
                    }
                },
                "arguments": {"type": "object"},
                "llm_extract_schema": {"type": "object"}
            },
            "required": ["plan_properties", "arguments", "llm_extract_schema"],
            "additionalProperties": false
        });
        let reason = unsupported_native_construct(&schema);
        assert!(reason.is_some());
    }

    #[test]
    fn allows_closed_schemas_including_empty_property_maps() {
        for schema in [
            json!({"type": "string"}),
            json!({"type": "object", "properties": {"name": {"type": "string"}}, "additionalProperties": false}),
            // An explicit empty `properties` map plus `additionalProperties:
            // false` is a valid closed schema (object must have no fields),
            // distinct from an object with no `properties` key at all.
            json!({"type": "object", "properties": {}, "additionalProperties": false}),
            json!({"type": "array", "items": {"type": "string"}}),
        ] {
            assert!(
                unsupported_native_construct(&schema).is_none(),
                "did not expect {schema} to be flagged"
            );
        }
    }

    #[test]
    fn flags_empty_properties_without_an_explicit_additional_properties_false() {
        // Per JSON Schema, `additionalProperties` defaults to `true` (open)
        // when absent - so `properties: {}` alone (no properties, no
        // explicit closing) is effectively free-form, not the same as
        // `properties: {}, additionalProperties: false` above.
        let schema = json!({"type": "object", "properties": {}});
        assert!(unsupported_native_construct(&schema).is_some());
    }

    #[test]
    fn flags_unsupported_construct_nested_under_a_non_object_root() {
        let schema = json!({"type": "array", "items": {}});
        assert!(unsupported_native_construct(&schema).is_some());
    }

    #[test]
    fn prompt_instruction_does_not_require_a_json_object_for_non_object_roots() {
        let instruction =
            schema_prompt_instruction(&json!({"type": "array", "items": {"type": "string"}}));
        let lower = instruction.to_lowercase();
        assert!(lower.contains("json value"));
        assert!(!lower.contains("json object"));
    }
}
