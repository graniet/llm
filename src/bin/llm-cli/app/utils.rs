use crate::provider::{parse_provider_string, ProviderId};

pub fn parse_provider_and_model(input: Option<&str>) -> Option<(ProviderId, Option<String>)> {
    parse_provider_string(input)
}
