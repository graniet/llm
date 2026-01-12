use super::llm_builder::LLMBuilder;

impl LLMBuilder {
    /// Enable web search for OpenAI-compatible providers.
    pub fn openai_enable_web_search(mut self, enable: bool) -> Self {
        self.state.openai_enable_web_search = Some(enable);
        self
    }

    /// Set the web search context size.
    pub fn openai_web_search_context_size(mut self, context_size: impl Into<String>) -> Self {
        self.state.openai_web_search_context_size = Some(context_size.into());
        self
    }

    /// Set the web search user location type.
    pub fn openai_web_search_user_location_type(
        mut self,
        location_type: impl Into<String>,
    ) -> Self {
        self.state.openai_web_search_user_location_type = Some(location_type.into());
        self
    }

    /// Set the web search user location approximate country.
    pub fn openai_web_search_user_location_approximate_country(
        mut self,
        country: impl Into<String>,
    ) -> Self {
        self.state
            .openai_web_search_user_location_approximate_country = Some(country.into());
        self
    }

    /// Set the web search user location approximate city.
    pub fn openai_web_search_user_location_approximate_city(
        mut self,
        city: impl Into<String>,
    ) -> Self {
        self.state.openai_web_search_user_location_approximate_city = Some(city.into());
        self
    }

    /// Set the web search user location approximate region.
    pub fn openai_web_search_user_location_approximate_region(
        mut self,
        region: impl Into<String>,
    ) -> Self {
        self.state
            .openai_web_search_user_location_approximate_region = Some(region.into());
        self
    }

    /// Backward compatible alias for xAI search mode.
    #[deprecated(note = "Renamed to `xai_search_mode`.")]
    pub fn search_mode(self, mode: impl Into<String>) -> Self {
        self.xai_search_mode(mode)
    }

    /// Sets the search mode for search-enabled providers.
    pub fn xai_search_mode(mut self, mode: impl Into<String>) -> Self {
        self.state.xai_search_mode = Some(mode.into());
        self
    }

    /// Adds a search source with optional excluded websites.
    pub fn xai_search_source(
        mut self,
        source_type: impl Into<String>,
        excluded_websites: Option<Vec<String>>,
    ) -> Self {
        self.state.xai_search_source_type = Some(source_type.into());
        self.state.xai_search_excluded_websites = excluded_websites;
        self
    }

    /// Sets the maximum number of search results.
    pub fn xai_max_search_results(mut self, max: u32) -> Self {
        self.state.xai_search_max_results = Some(max);
        self
    }

    /// Sets the date range for search results.
    pub fn xai_search_date_range(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        self.state.xai_search_from_date = Some(from.into());
        self.state.xai_search_to_date = Some(to.into());
        self
    }

    /// Sets the start date for search results (format: "YYYY-MM-DD").
    pub fn xai_search_from_date(mut self, date: impl Into<String>) -> Self {
        self.state.xai_search_from_date = Some(date.into());
        self
    }

    /// Sets the end date for search results (format: "YYYY-MM-DD").
    pub fn xai_search_to_date(mut self, date: impl Into<String>) -> Self {
        self.state.xai_search_to_date = Some(date.into());
        self
    }
}
