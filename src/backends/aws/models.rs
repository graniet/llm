// src/backends/bedrock/models.rs
//! AWS Bedrock model definitions and capabilities

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Supported AWS Bedrock models
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(untagged)]
pub enum BedrockModel {
    /// Direct model access (standard model IDs)
    Direct(DirectModel),

    /// Cross-region inference profile
    CrossRegion {
        region: String,
        model: CrossRegionModel,
    },

    /// Custom model ID or ARN
    Custom(String),
}

/// Direct model access variants
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DirectModel {
    // Claude models
    #[serde(rename = "anthropic.claude-3-5-sonnet-20241022-v2:0")]
    ClaudeSonnet35V2,

    #[serde(rename = "us.anthropic.claude-sonnet-4-0-v1:0")]
    ClaudeSonnet4,

    #[serde(rename = "anthropic.claude-3-opus-20240229-v1:0")]
    ClaudeOpus3,

    #[serde(rename = "anthropic.claude-3-5-sonnet-20240620-v1:0")]
    ClaudeSonnet35,

    #[serde(rename = "anthropic.claude-3-sonnet-20240229-v1:0")]
    ClaudeSonnet3,

    #[serde(rename = "anthropic.claude-3-haiku-20240307-v1:0")]
    ClaudeHaiku3,

    // Llama models
    #[serde(rename = "meta.llama3-2-90b-instruct-v1:0")]
    Llama32_90B,

    #[serde(rename = "meta.llama3-2-11b-instruct-v1:0")]
    Llama32_11B,

    #[serde(rename = "meta.llama3-2-3b-instruct-v1:0")]
    Llama32_3B,

    #[serde(rename = "meta.llama3-2-1b-instruct-v1:0")]
    Llama32_1B,

    #[serde(rename = "meta.llama3-1-70b-instruct-v1:0")]
    Llama31_70B,

    #[serde(rename = "meta.llama3-1-8b-instruct-v1:0")]
    Llama31_8B,

    // Amazon Titan models
    #[serde(rename = "amazon.titan-text-premier-v1:0")]
    TitanTextPremier,

    #[serde(rename = "amazon.titan-text-express-v1")]
    TitanTextExpress,

    #[serde(rename = "amazon.titan-text-lite-v1")]
    TitanTextLite,

    #[serde(rename = "amazon.titan-embed-text-v2:0")]
    TitanEmbedV2,

    #[serde(rename = "amazon.titan-embed-text-v1")]
    TitanEmbedV1,

    // Cohere models
    #[serde(rename = "cohere.command-r-plus-v1:0")]
    CohereCommandRPlus,

    #[serde(rename = "cohere.command-r-v1:0")]
    CohereCommandR,

    #[serde(rename = "cohere.embed-english-v3")]
    CohereEmbedV3,

    #[serde(rename = "cohere.embed-multilingual-v3")]
    CohereEmbedMultilingualV3,

    // Mistral models
    #[serde(rename = "mistral.mistral-large-2407-v1:0")]
    MistralLarge,

    #[serde(rename = "mistral.mistral-small-2402-v1:0")]
    MistralSmall,
}

/// Cross-region inference profile models
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CrossRegionModel {
    // Claude models
    #[serde(rename = "claude-sonnet-4-20250514-v1:0")]
    ClaudeSonnet4,

    #[serde(rename = "claude-sonnet-4-5-20250929-v1:0")]
    ClaudeSonnet45,

    #[serde(rename = "claude-3-5-sonnet-20241022-v2:0")]
    ClaudeSonnet35V2,

    #[serde(rename = "claude-3-opus-20240229-v1:0")]
    ClaudeOpus3,

    #[serde(rename = "claude-3-5-sonnet-20240620-v1:0")]
    ClaudeSonnet35,

    #[serde(rename = "claude-3-haiku-20240307-v1:0")]
    ClaudeHaiku3,

    // Mistral models
    #[serde(rename = "pixtral-large-2502-v1:0")]
    MistralPixtralLarge,

    // Cohere models
    #[serde(rename = "embed-v4:0")]
    CohereEmbedV4,
}

impl BedrockModel {
    /// Create a cross-region inference profile model
    ///
    /// # Examples
    /// ```
    /// use llm::backends::aws::*;
    ///
    /// // Use Claude Sonnet 4 from EU region
    /// let model = BedrockModel::cross_region("eu-central-1", CrossRegionModel::ClaudeSonnet4);
    ///
    /// // Use Mistral Pixtral from EU
    /// let model = BedrockModel::cross_region("eu-central-1", CrossRegionModel::MistralPixtralLarge);
    /// ```
    pub fn cross_region(region: impl Into<String>, model: CrossRegionModel) -> Self {
        Self::CrossRegion {
            region: region.into(),
            model,
        }
    }

    /// Convenient shorthand for EU cross-region models
    ///
    /// # Examples
    /// ```
    /// use llm::backends::aws::*;
    ///
    /// let model = BedrockModel::eu(CrossRegionModel::ClaudeSonnet4);
    /// ```
    pub fn eu(model: CrossRegionModel) -> Self {
        Self::cross_region("eu-central-1", model)
    }

    /// Convenient shorthand for US cross-region models
    pub fn us(model: CrossRegionModel) -> Self {
        Self::cross_region("us-east-1", model)
    }

    /// Get the model ID string used by AWS Bedrock
    ///
    /// This can be either a simple model ID or a full ARN for cross-region inference profiles
    pub fn model_id(&self) -> String {
        match self {
            Self::Direct(model) => model.model_id().to_string(),
            Self::CrossRegion { region, model } => {
                let vendor = model.vendor();
                let model_name = model.model_id();
                let region_prefix = Self::region_prefix(region);
                format!(
                    "arn:aws:bedrock:{}::inference-profile/{}.{}.{}",
                    region, region_prefix, vendor, model_name
                )
            }
            Self::Custom(id) => id.clone(),
        }
    }

    fn region_prefix(region: &str) -> &str {
        match region {
            "us-east-1" => "us",
            "us-west-2" => "us",
            "eu-central-1" => "eu",
            "eu-west-1" => "eu",
            "eu-west-2" => "eu",
            "ap-northeast-1" => "ap",
            "ap-southeast-1" => "ap",
            "ap-southeast-2" => "ap",
            _ => region,
        }
    }

    /// Create a BedrockModel from a custom model ID or ARN
    pub fn from_id(id: impl Into<String>) -> Self {
        let id = id.into();

        // Check if it's an ARN
        if id.starts_with("arn:aws:bedrock") {
            if let Some(cross_region) = Self::parse_cross_region_arn(&id) {
                return cross_region;
            }
        }

        // Try to match against direct models
        if let Some(direct) = DirectModel::from_id(&id) {
            return Self::Direct(direct);
        }

        // Otherwise treat as custom
        Self::Custom(id)
    }

    fn parse_cross_region_arn(arn: &str) -> Option<Self> {
        // ARN format: arn:aws:bedrock:region:account:inference-profile/region-prefix.vendor.model-id
        let parts: Vec<&str> = arn.split(':').collect();
        if parts.len() < 6 {
            return None;
        }

        let region = parts[3];
        let profile_part = parts.get(5)?;

        // Extract model info from inference-profile/region-prefix.vendor.model-id
        let profile_info = profile_part.strip_prefix("inference-profile/")?;
        let info_parts: Vec<&str> = profile_info.splitn(3, '.').collect();
        if info_parts.len() < 3 {
            return None;
        }

        CrossRegionModel::from_vendor_and_id(info_parts[1], info_parts[2]).map(|cross_model| Self::CrossRegion {
                region: region.to_string(),
                model: cross_model,
            })
    }

    /// Check if this is a cross-region inference profile (ARN-based)
    pub fn is_cross_region_profile(&self) -> bool {
        matches!(self, Self::CrossRegion { .. })
    }

    /// Get the underlying model (Direct or CrossRegion) for capability checks
    fn inner_model(&self) -> InnerModel {
        match self {
            Self::Direct(model) => InnerModel::Direct(*model),
            Self::CrossRegion { model, .. } => InnerModel::CrossRegion(*model),
            Self::Custom(id) => {
                // Try to infer from custom ID
                if id.contains("embed") {
                    InnerModel::Embedding
                } else {
                    InnerModel::Chat
                }
            }
        }
    }

    pub(crate) fn override_keys(&self) -> Vec<String> {
        let mut keys = vec![self.model_id()];

        if let Self::CrossRegion { region, model } = self {
            keys.push(format!(
                "{}.{}.{}",
                Self::region_prefix(region),
                model.vendor(),
                model.model_id()
            ));
            keys.push(format!("{}.{}", model.vendor(), model.model_id()));
            keys.push(model.model_id().to_string());
        }

        keys
    }
}

impl DirectModel {
    fn model_id(&self) -> &str {
        match self {
            Self::ClaudeSonnet35V2 => "anthropic.claude-3-5-sonnet-20241022-v2:0",
            Self::ClaudeSonnet4 => "us.anthropic.claude-sonnet-4-0-v1:0",
            Self::ClaudeOpus3 => "anthropic.claude-3-opus-20240229-v1:0",
            Self::ClaudeSonnet35 => "anthropic.claude-3-5-sonnet-20240620-v1:0",
            Self::ClaudeSonnet3 => "anthropic.claude-3-sonnet-20240229-v1:0",
            Self::ClaudeHaiku3 => "anthropic.claude-3-haiku-20240307-v1:0",
            Self::Llama32_90B => "meta.llama3-2-90b-instruct-v1:0",
            Self::Llama32_11B => "meta.llama3-2-11b-instruct-v1:0",
            Self::Llama32_3B => "meta.llama3-2-3b-instruct-v1:0",
            Self::Llama32_1B => "meta.llama3-2-1b-instruct-v1:0",
            Self::Llama31_70B => "meta.llama3-1-70b-instruct-v1:0",
            Self::Llama31_8B => "meta.llama3-1-8b-instruct-v1:0",
            Self::TitanTextPremier => "amazon.titan-text-premier-v1:0",
            Self::TitanTextExpress => "amazon.titan-text-express-v1",
            Self::TitanTextLite => "amazon.titan-text-lite-v1",
            Self::TitanEmbedV2 => "amazon.titan-embed-text-v2:0",
            Self::TitanEmbedV1 => "amazon.titan-embed-text-v1",
            Self::CohereCommandRPlus => "cohere.command-r-plus-v1:0",
            Self::CohereCommandR => "cohere.command-r-v1:0",
            Self::CohereEmbedV3 => "cohere.embed-english-v3",
            Self::CohereEmbedMultilingualV3 => "cohere.embed-multilingual-v3",
            Self::MistralLarge => "mistral.mistral-large-2407-v1:0",
            Self::MistralSmall => "mistral.mistral-small-2402-v1:0",
        }
    }

    fn from_id(id: &str) -> Option<Self> {
        match id {
            "anthropic.claude-3-5-sonnet-20241022-v2:0" => Some(Self::ClaudeSonnet35V2),
            "us.anthropic.claude-sonnet-4-0-v1:0" => Some(Self::ClaudeSonnet4),
            "anthropic.claude-3-opus-20240229-v1:0" => Some(Self::ClaudeOpus3),
            "anthropic.claude-3-5-sonnet-20240620-v1:0" => Some(Self::ClaudeSonnet35),
            "anthropic.claude-3-sonnet-20240229-v1:0" => Some(Self::ClaudeSonnet3),
            "anthropic.claude-3-haiku-20240307-v1:0" => Some(Self::ClaudeHaiku3),
            "meta.llama3-2-90b-instruct-v1:0" => Some(Self::Llama32_90B),
            "meta.llama3-2-11b-instruct-v1:0" => Some(Self::Llama32_11B),
            "meta.llama3-2-3b-instruct-v1:0" => Some(Self::Llama32_3B),
            "meta.llama3-2-1b-instruct-v1:0" => Some(Self::Llama32_1B),
            "meta.llama3-1-70b-instruct-v1:0" => Some(Self::Llama31_70B),
            "meta.llama3-1-8b-instruct-v1:0" => Some(Self::Llama31_8B),
            "amazon.titan-text-premier-v1:0" => Some(Self::TitanTextPremier),
            "amazon.titan-text-express-v1" => Some(Self::TitanTextExpress),
            "amazon.titan-text-lite-v1" => Some(Self::TitanTextLite),
            "amazon.titan-embed-text-v2:0" => Some(Self::TitanEmbedV2),
            "amazon.titan-embed-text-v1" => Some(Self::TitanEmbedV1),
            "cohere.command-r-plus-v1:0" => Some(Self::CohereCommandRPlus),
            "cohere.command-r-v1:0" => Some(Self::CohereCommandR),
            "cohere.embed-english-v3" => Some(Self::CohereEmbedV3),
            "cohere.embed-multilingual-v3" => Some(Self::CohereEmbedMultilingualV3),
            "mistral.mistral-large-2407-v1:0" => Some(Self::MistralLarge),
            "mistral.mistral-small-2402-v1:0" => Some(Self::MistralSmall),
            _ => None,
        }
    }
}

impl CrossRegionModel {
    fn model_id(&self) -> &str {
        match self {
            Self::ClaudeSonnet4 => "claude-sonnet-4-20250514-v1:0",
            Self::ClaudeSonnet45 => "claude-sonnet-4-5-20250929-v1:0",
            Self::ClaudeSonnet35V2 => "claude-3-5-sonnet-20241022-v2:0",
            Self::ClaudeOpus3 => "claude-3-opus-20240229-v1:0",
            Self::ClaudeSonnet35 => "claude-3-5-sonnet-20240620-v1:0",
            Self::ClaudeHaiku3 => "claude-3-haiku-20240307-v1:0",
            Self::MistralPixtralLarge => "pixtral-large-2502-v1:0",
            Self::CohereEmbedV4 => "embed-v4:0",
        }
    }

    fn vendor(&self) -> &str {
        match self {
            Self::ClaudeSonnet4
            | Self::ClaudeSonnet45
            | Self::ClaudeSonnet35V2
            | Self::ClaudeOpus3
            | Self::ClaudeSonnet35
            | Self::ClaudeHaiku3 => "anthropic",
            Self::MistralPixtralLarge => "mistral",
            Self::CohereEmbedV4 => "cohere",
        }
    }

    fn from_vendor_and_id(vendor: &str, model_id: &str) -> Option<Self> {
        match (vendor, model_id) {
            ("anthropic", id) if id.contains("claude-sonnet-4-20250514") => {
                Some(Self::ClaudeSonnet4)
            }
            ("anthropic", id) if id.contains("claude-sonnet-4-5-20250929") => {
                Some(Self::ClaudeSonnet45)
            }
            ("anthropic", id) if id.contains("claude-3-5-sonnet-20241022") => {
                Some(Self::ClaudeSonnet35V2)
            }
            ("anthropic", id) if id.contains("claude-3-opus") => Some(Self::ClaudeOpus3),
            ("anthropic", id) if id.contains("claude-3-5-sonnet-20240620") => {
                Some(Self::ClaudeSonnet35)
            }
            ("anthropic", id) if id.contains("claude-3-haiku") => Some(Self::ClaudeHaiku3),
            ("mistral", id) if id.contains("pixtral-large") => Some(Self::MistralPixtralLarge),
            ("cohere", id) if id.contains("embed-v4") => Some(Self::CohereEmbedV4),
            _ => None,
        }
    }
}

// Helper enum for capability checks
#[derive(Debug, Clone, Copy)]
enum InnerModel {
    Direct(DirectModel),
    CrossRegion(CrossRegionModel),
    Chat,
    Embedding,
}

impl BedrockModel {
    /// Check if this model supports a specific capability
    pub fn supports(&self, capability: ModelCapability) -> bool {
        match capability {
            ModelCapability::Completion => self.is_text_model(),
            ModelCapability::Chat => self.is_chat_model(),
            ModelCapability::Embeddings => self.is_embedding_model(),
            ModelCapability::Vision => self.supports_vision_impl(),
            ModelCapability::ToolUse => self.supports_tools_impl(),
            ModelCapability::Streaming => self.is_text_model() || self.is_chat_model(),
        }
    }

    fn is_text_model(&self) -> bool {
        !self.is_embedding_model()
    }

    fn is_chat_model(&self) -> bool {
        match self.inner_model() {
            InnerModel::Direct(model) => model.is_chat_model(),
            InnerModel::CrossRegion(model) => model.is_chat_model(),
            InnerModel::Chat => true,
            InnerModel::Embedding => false,
        }
    }

    fn is_embedding_model(&self) -> bool {
        match self.inner_model() {
            InnerModel::Direct(model) => model.is_embedding_model(),
            InnerModel::CrossRegion(model) => model.is_embedding_model(),
            InnerModel::Chat => false,
            InnerModel::Embedding => true,
        }
    }

    fn supports_vision_impl(&self) -> bool {
        match self.inner_model() {
            InnerModel::Direct(model) => model.supports_vision(),
            InnerModel::CrossRegion(model) => model.supports_vision(),
            _ => false,
        }
    }

    fn supports_tools_impl(&self) -> bool {
        match self.inner_model() {
            InnerModel::Direct(model) => model.supports_tools(),
            InnerModel::CrossRegion(model) => model.supports_tools(),
            _ => false,
        }
    }

    /// Get the maximum tokens this model can handle in output
    pub fn max_output_tokens(&self) -> u32 {
        match self.inner_model() {
            InnerModel::Direct(model) => model.max_output_tokens(),
            InnerModel::CrossRegion(model) => model.max_output_tokens(),
            _ => 4096,
        }
    }

    /// Get context window size for this model
    pub fn context_window(&self) -> u32 {
        match self.inner_model() {
            InnerModel::Direct(model) => model.context_window(),
            InnerModel::CrossRegion(model) => model.context_window(),
            _ => 128_000,
        }
    }
}

impl DirectModel {
    fn is_chat_model(&self) -> bool {
        !matches!(
            self,
            Self::TitanEmbedV2
                | Self::TitanEmbedV1
                | Self::CohereEmbedV3
                | Self::CohereEmbedMultilingualV3
        )
    }

    fn is_embedding_model(&self) -> bool {
        matches!(
            self,
            Self::TitanEmbedV2
                | Self::TitanEmbedV1
                | Self::CohereEmbedV3
                | Self::CohereEmbedMultilingualV3
        )
    }

    fn supports_vision(&self) -> bool {
        matches!(
            self,
            Self::ClaudeSonnet35V2
                | Self::ClaudeSonnet4
                | Self::ClaudeOpus3
                | Self::ClaudeSonnet35
                | Self::ClaudeSonnet3
                | Self::ClaudeHaiku3
                | Self::Llama32_90B
                | Self::Llama32_11B
        )
    }

    fn supports_tools(&self) -> bool {
        matches!(
            self,
            Self::ClaudeSonnet35V2
                | Self::ClaudeSonnet4
                | Self::ClaudeOpus3
                | Self::ClaudeSonnet35
                | Self::ClaudeSonnet3
                | Self::ClaudeHaiku3
                | Self::CohereCommandRPlus
                | Self::CohereCommandR
                | Self::MistralLarge
        )
    }

    fn max_output_tokens(&self) -> u32 {
        match self {
            Self::ClaudeSonnet35V2 | Self::ClaudeSonnet4 => 8192,
            Self::ClaudeOpus3 | Self::ClaudeSonnet35 => 8192,
            Self::ClaudeSonnet3 | Self::ClaudeHaiku3 => 4096,
            Self::Llama32_90B | Self::Llama32_11B | Self::Llama31_70B => 4096,
            Self::Llama32_3B | Self::Llama32_1B | Self::Llama31_8B => 2048,
            Self::TitanTextPremier | Self::TitanTextExpress => 8192,
            Self::TitanTextLite => 4096,
            Self::CohereCommandRPlus | Self::CohereCommandR => 4096,
            Self::MistralLarge | Self::MistralSmall => 8192,
            _ => 0,
        }
    }

    fn context_window(&self) -> u32 {
        match self {
            Self::ClaudeSonnet35V2 | Self::ClaudeSonnet4 => 200_000,
            Self::ClaudeOpus3 | Self::ClaudeSonnet35 | Self::ClaudeSonnet3 | Self::ClaudeHaiku3 => {
                200_000
            }
            Self::Llama32_90B | Self::Llama32_11B => 128_000,
            Self::Llama32_3B | Self::Llama32_1B => 128_000,
            Self::Llama31_70B | Self::Llama31_8B => 128_000,
            Self::TitanTextPremier => 32_000,
            Self::TitanTextExpress | Self::TitanTextLite => 8_000,
            Self::CohereCommandRPlus | Self::CohereCommandR => 128_000,
            Self::MistralLarge | Self::MistralSmall => 128_000,
            _ => 0,
        }
    }
}

impl CrossRegionModel {
    fn is_chat_model(&self) -> bool {
        !matches!(self, Self::CohereEmbedV4)
    }

    fn is_embedding_model(&self) -> bool {
        matches!(self, Self::CohereEmbedV4)
    }

    fn supports_vision(&self) -> bool {
        matches!(
            self,
            Self::ClaudeSonnet4
                | Self::ClaudeSonnet45
                | Self::ClaudeSonnet35V2
                | Self::ClaudeOpus3
                | Self::ClaudeSonnet35
                | Self::ClaudeHaiku3
                | Self::MistralPixtralLarge
        )
    }

    fn supports_tools(&self) -> bool {
        matches!(
            self,
            Self::ClaudeSonnet4
                | Self::ClaudeSonnet45
                | Self::ClaudeSonnet35V2
                | Self::ClaudeOpus3
                | Self::ClaudeSonnet35
                | Self::ClaudeHaiku3
                | Self::MistralPixtralLarge
        )
    }

    fn max_output_tokens(&self) -> u32 {
        match self {
            Self::ClaudeSonnet4 | Self::ClaudeSonnet45 | Self::ClaudeSonnet35V2 => 8192,
            Self::ClaudeOpus3 | Self::ClaudeSonnet35 => 8192,
            Self::ClaudeHaiku3 => 4096,
            Self::MistralPixtralLarge => 8192,
            Self::CohereEmbedV4 => 0,
        }
    }

    fn context_window(&self) -> u32 {
        match self {
            Self::ClaudeSonnet4 | Self::ClaudeSonnet45 | Self::ClaudeSonnet35V2 => 200_000,
            Self::ClaudeOpus3 | Self::ClaudeSonnet35 | Self::ClaudeHaiku3 => 200_000,
            Self::MistralPixtralLarge => 128_000,
            Self::CohereEmbedV4 => 0,
        }
    }
}

impl fmt::Display for BedrockModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.model_id())
    }
}

impl fmt::Display for DirectModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.model_id())
    }
}

impl fmt::Display for CrossRegionModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.model_id())
    }
}

/// Model capabilities
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelCapability {
    /// Text completion
    Completion,

    /// Chat/conversation
    Chat,

    /// Text embeddings
    Embeddings,

    /// Vision (image understanding)
    Vision,

    /// Tool/function calling
    ToolUse,

    /// Streaming responses
    Streaming,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelCapabilityOverrides {
    #[serde(default)]
    pub models: HashMap<String, ModelCapabilityOverride>,
    #[serde(default)]
    pub model: Vec<NamedModelCapabilityOverride>,
}

impl ModelCapabilityOverrides {
    pub(crate) fn supports(
        &self,
        model: &BedrockModel,
        capability: ModelCapability,
    ) -> Option<bool> {
        let keys = model.override_keys();
        let normalized_keys: Vec<String> = keys
            .iter()
            .map(|key| Self::normalize_bedrock_arn(key).unwrap_or_else(|| key.clone()))
            .collect();

        for key in &keys {
            if let Some(override_entry) = self.models.get(key.as_str()) {
                if let Some(supports) = override_entry.supports(capability) {
                    return Some(supports);
                }
            }
        }

        for (key, override_entry) in &self.models {
            let normalized_key = Self::normalize_bedrock_arn(key).unwrap_or_else(|| key.clone());
            if normalized_keys.contains(&normalized_key) {
                if let Some(supports) = override_entry.supports(capability) {
                    return Some(supports);
                }
            }
        }

        for override_entry in &self.model {
            if !keys.contains(&override_entry.name) {
                let normalized_name = Self::normalize_bedrock_arn(&override_entry.name)
                    .unwrap_or_else(|| override_entry.name.clone());
                if !normalized_keys.contains(&normalized_name) {
                    continue;
                }
            }

            if let Some(supports) = override_entry.overrides.supports(capability) {
                return Some(supports);
            }
        }

        None
    }

    fn normalize_bedrock_arn(value: &str) -> Option<String> {
        let parts: Vec<&str> = value.splitn(6, ':').collect();
        if parts.len() != 6 {
            return None;
        }
        if parts[0] != "arn" || parts[2] != "bedrock" {
            return None;
        }

        let mut normalized: Vec<String> = parts.iter().map(|part| (*part).to_string()).collect();
        normalized[4].clear();
        Some(normalized.join(":"))
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NamedModelCapabilityOverride {
    pub name: String,
    #[serde(flatten)]
    pub overrides: ModelCapabilityOverride,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelCapabilityOverride {
    #[serde(default)]
    pub completion: Option<bool>,
    #[serde(default)]
    pub chat: Option<bool>,
    #[serde(default)]
    pub embeddings: Option<bool>,
    #[serde(default)]
    pub vision: Option<bool>,
    #[serde(default)]
    pub tool_use: Option<bool>,
    #[serde(default)]
    pub streaming: Option<bool>,
}

impl ModelCapabilityOverride {
    fn supports(&self, capability: ModelCapability) -> Option<bool> {
        match capability {
            ModelCapability::Completion => self.completion,
            ModelCapability::Chat => self.chat,
            ModelCapability::Embeddings => self.embeddings,
            ModelCapability::Vision => self.vision,
            ModelCapability::ToolUse => self.tool_use,
            ModelCapability::Streaming => self.streaming,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_model_id() {
            let model = BedrockModel::Direct(DirectModel::ClaudeSonnet4);
            assert_eq!(model.model_id(), "us.anthropic.claude-sonnet-4-0-v1:0");

            // Test cross-region inference profile
            let model = BedrockModel::eu(CrossRegionModel::ClaudeSonnet4);
            assert!(model.model_id().starts_with("arn:aws:bedrock:eu-central-1"));
            assert!(model.model_id().contains("claude-sonnet-4-20250514"));

            let model =
                BedrockModel::cross_region("eu-central-1", CrossRegionModel::MistralPixtralLarge);
            assert!(model.model_id().contains("pixtral-large"));

            let model = BedrockModel::eu(CrossRegionModel::CohereEmbedV4);
            assert!(model.model_id().contains("embed-v4"));
        }

        #[test]
        fn test_cross_region_convenience_methods() {
            // Test EU convenience method
            let model = BedrockModel::eu(CrossRegionModel::ClaudeSonnet4);
            assert!(
                matches!(model, BedrockModel::CrossRegion { region, .. } if region == "eu-central-1")
            );

            // Test US convenience method
            let model = BedrockModel::us(CrossRegionModel::ClaudeSonnet4);
            assert!(
                matches!(model, BedrockModel::CrossRegion { region, .. } if region == "us-east-1")
            );

            // Test custom region
            let model =
                BedrockModel::cross_region("ap-southeast-1", CrossRegionModel::ClaudeSonnet4);
            assert!(
                matches!(model, BedrockModel::CrossRegion { region, .. } if region == "ap-southeast-1")
            );
        }

        #[test]
        fn test_from_id() {
            // Test standard model IDs
            let model = BedrockModel::from_id("us.anthropic.claude-sonnet-4-0-v1:0");
            assert!(matches!(
                model,
                BedrockModel::Direct(DirectModel::ClaudeSonnet4)
            ));

            // Test ARN parsing
            let model = BedrockModel::from_id(
            "arn:aws:bedrock:eu-central-1:876164100382:inference-profile/eu.anthropic.claude-sonnet-4-20250514-v1:0"
        );
            assert!(matches!(model, BedrockModel::CrossRegion { .. }));

            // Test custom model
            let model = BedrockModel::from_id("custom-model-id");
            assert!(matches!(model, BedrockModel::Custom(_)));
        }

        #[test]
        fn test_is_cross_region_profile() {
            let direct = BedrockModel::Direct(DirectModel::ClaudeSonnet4);
            assert!(!direct.is_cross_region_profile());

            let cross_region = BedrockModel::eu(CrossRegionModel::ClaudeSonnet4);
            assert!(cross_region.is_cross_region_profile());

            let pixtral = BedrockModel::eu(CrossRegionModel::MistralPixtralLarge);
            assert!(pixtral.is_cross_region_profile());
        }

        #[test]
        fn test_model_capabilities() {
            let claude = BedrockModel::Direct(DirectModel::ClaudeSonnet4);
            assert!(claude.supports(ModelCapability::Chat));
            assert!(claude.supports(ModelCapability::Vision));
            assert!(claude.supports(ModelCapability::ToolUse));
            assert!(!claude.supports(ModelCapability::Embeddings));

            // Test EU cross-region models
            let claude_eu = BedrockModel::eu(CrossRegionModel::ClaudeSonnet4);
            assert!(claude_eu.supports(ModelCapability::Chat));
            assert!(claude_eu.supports(ModelCapability::Vision));
            assert!(claude_eu.supports(ModelCapability::ToolUse));

            let pixtral = BedrockModel::eu(CrossRegionModel::MistralPixtralLarge);
            assert!(pixtral.supports(ModelCapability::Chat));
            assert!(pixtral.supports(ModelCapability::Vision));
            assert!(pixtral.supports(ModelCapability::ToolUse));

            let titan_embed = BedrockModel::Direct(DirectModel::TitanEmbedV2);
            assert!(titan_embed.supports(ModelCapability::Embeddings));
            assert!(!titan_embed.supports(ModelCapability::Chat));

            let cohere_embed_eu = BedrockModel::eu(CrossRegionModel::CohereEmbedV4);
            assert!(cohere_embed_eu.supports(ModelCapability::Embeddings));
            assert!(!cohere_embed_eu.supports(ModelCapability::Chat));
        }

        #[test]
        fn test_context_window() {
            let claude = BedrockModel::Direct(DirectModel::ClaudeSonnet4);
            assert_eq!(claude.context_window(), 200_000);

            let claude_eu = BedrockModel::eu(CrossRegionModel::ClaudeSonnet4);
            assert_eq!(claude_eu.context_window(), 200_000);

            let llama = BedrockModel::Direct(DirectModel::Llama32_90B);
            assert_eq!(llama.context_window(), 128_000);

            let pixtral = BedrockModel::eu(CrossRegionModel::MistralPixtralLarge);
            assert_eq!(pixtral.context_window(), 128_000);
        }

        #[test]
        fn test_max_output_tokens() {
            let claude = BedrockModel::Direct(DirectModel::ClaudeSonnet4);
            assert_eq!(claude.max_output_tokens(), 8192);

            let claude_eu = BedrockModel::eu(CrossRegionModel::ClaudeSonnet45);
            assert_eq!(claude_eu.max_output_tokens(), 8192);

            let llama = BedrockModel::Direct(DirectModel::Llama32_3B);
            assert_eq!(llama.max_output_tokens(), 2048);
        }
    }
}
