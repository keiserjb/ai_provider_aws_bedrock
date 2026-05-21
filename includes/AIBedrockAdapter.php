<?php

/**
 * @file
 * AWS Bedrock adapter for the AI module.
 *
 * This adapter implements AIProviderClient using the AWS SDK for PHP,
 * translating shared chat-style method calls to AWS Bedrock's Converse API.
 *
 * Supports chat via the Converse API and embeddings via InvokeModel.
 * Models available through Bedrock include Anthropic Claude, Meta Llama,
 * Mistral, Amazon Titan, Cohere, and more.
 *
 * @see https://docs.aws.amazon.com/bedrock/latest/userguide/
 */
class AIBedrockAdapter extends AIAdapterBase {

  /**
   * AWS Access Key ID (mirrors $this->apiKey from base).
   *
   * @var string
   */
  protected $accessKeyId;

  /**
   * AWS Secret Access Key.
   *
   * @var string
   */
  protected $secretAccessKey;

  /**
   * AWS region.
   *
   * @var string
   */
  protected $region;

  /**
   * The Bedrock Runtime client.
   *
   * @var \Aws\BedrockRuntime\BedrockRuntimeClient|null
   */
  protected $runtimeClient = NULL;

  /**
   * The Bedrock client (for listing models).
   *
   * @var \Aws\Bedrock\BedrockClient|null
   */
  protected $bedrockClient = NULL;

  /**
   * Constructor.
   *
   * @param string         $api_key
   *   The AWS Access Key ID (resolved from the Key module).
   * @param AIApi|null $api
   *   Optional AIApi wrapper for logging.
   */
  public function __construct($api_key, ?AIApi $api = NULL) {
    parent::__construct($api_key, $api);
    $this->accessKeyId = $this->apiKey;

    if (empty($this->accessKeyId)) {
      throw new \Exception('AWS Access Key ID is required');
    }

    // Load the secret key from the providers config.
    $secret_key_name = ai_get_provider_setting('aws_bedrock', 'aws_secret_key', '');

    if (!empty($secret_key_name) && function_exists('key_get_key_value')) {
      $this->secretAccessKey = trim(key_get_key_value($secret_key_name));
    }

    if (empty($this->secretAccessKey)) {
      throw new \Exception(
        'AWS Secret Access Key is required. Configure it in the AI provider settings.'
      );
    }

    // Prefer the shared AI provider config and fall back to the legacy
    // module config so existing installs keep working after upgrade.
    $legacy_region = config_get('ai_provider_aws_bedrock.settings', 'aws_region');
    $this->region = ai_get_provider_setting('aws_bedrock', 'aws_region')
      ?: ($legacy_region ?: 'us-east-1');

    // Ensure the AWS SDK autoloader is available.
    $this->loadAwsSdk();
  }

  /**
   * {@inheritdoc}
   *
   * Bedrock uses the AWS SDK (SigV4 signing), not backdrop_http_request.
   * Return empty headers; SDK handles auth.
   */
  protected function getDefaultHeaders(): array {
    return [];
  }

  /**
   * Load the AWS SDK autoloader.
   *
   * Prefers Composer Manager's global autoloader if available, then falls back
   * to the vendor directory bundled with this module.
   */
  protected function loadAwsSdk() {
    if (class_exists('Aws\\Sdk')) {
      return;
    }

    if (module_exists('composer_manager') && function_exists('composer_manager_register_autoloader')) {
      composer_manager_register_autoloader();
      if (class_exists('Aws\\Sdk')) {
        return;
      }
    }

    $autoload = BACKDROP_ROOT . '/' . backdrop_get_path('module', 'ai_provider_aws_bedrock') . '/vendor/autoload.php';
    if (file_exists($autoload)) {
      require_once $autoload;
      return;
    }

    throw new \Exception(
      'AWS SDK not found. Run "composer install" in the ai_provider_aws_bedrock module directory, or enable the Composer Manager module.'
    );
  }

  /**
   * Get the Bedrock Runtime client.
   *
   * @return \Aws\BedrockRuntime\BedrockRuntimeClient
   *   The runtime client for invoking models.
   */
  protected function getRuntimeClient() {
    if (!$this->runtimeClient) {
      $this->runtimeClient = new \Aws\BedrockRuntime\BedrockRuntimeClient([
        'version'     => 'latest',
        'region'      => $this->region,
        'credentials' => [
          'key'    => $this->accessKeyId,
          'secret' => $this->secretAccessKey,
        ],
      ]);
    }
    return $this->runtimeClient;
  }

  /**
   * Get the Bedrock client (for listing models).
   *
   * @return \Aws\Bedrock\BedrockClient
   *   The client for model management operations.
   */
  protected function getBedrockClient() {
    if (!$this->bedrockClient) {
      $this->bedrockClient = new \Aws\Bedrock\BedrockClient([
        'version'     => 'latest',
        'region'      => $this->region,
        'credentials' => [
          'key'    => $this->accessKeyId,
          'secret' => $this->secretAccessKey,
        ],
      ]);
    }
    return $this->bedrockClient;
  }

  /**
   * {@inheritdoc}
   *
   * Builds the model list using a two-pass strategy:
   * 1. Fetch inference profiles for the configured region — these are the
   *    preferred model identifiers because many newer models *require* them.
   * 2. Fetch foundation models and add only those that are NOT already
   *    covered by an inference profile, filtering out models that are not
   *    usable for chat or embeddings (image generators, video, audio,
   *    rerank, etc.).
   */
  public function getModels(): array {
    try {
      $client = $this->getBedrockClient();
      $models = [];

      // Track which base model IDs are covered by inference profiles so we
      // can skip them when processing foundation models.
      $profile_base_ids = [];

      // ---- Pass 1: Inference profiles (preferred) ----
      // Only include profiles matching the configured region prefix or
      // 'global.' profiles.
      try {
        $region_prefix = $this->getRegionPrefix();
        $profiles_result = $client->listInferenceProfiles([]);
        if (!empty($profiles_result['inferenceProfileSummaries'])) {
          foreach ($profiles_result['inferenceProfileSummaries'] as $profile) {
            if (($profile['status'] ?? '') !== 'ACTIVE') {
              continue;
            }
            $id = $profile['inferenceProfileId'] ?? '';
            if (empty($id)) {
              continue;
            }

            // Filter by region prefix.
            if (preg_match('/^(us|eu|ap|global)\./', $id, $matches)) {
              $prefix = $matches[1] . '.';
              if ($prefix !== $region_prefix && $prefix !== 'global.') {
                continue;
              }
            }

            // Skip non-text profiles (image, video, audio models).
            if ($this->isNonTextModel($id)) {
              continue;
            }

            $name = !empty($profile['inferenceProfileName'])
              ? $profile['inferenceProfileName'] . ' (' . $id . ')'
              : $id;
            $models[$id] = $name;

            // Record the base model ID so we skip it in pass 2.
            $base_id = preg_replace('/^(us|eu|ap|global)\./', '', $id);
            $profile_base_ids[$base_id] = TRUE;
          }
        }
      }
      catch (\Exception $e) {
        watchdog(
          'ai_provider_aws_bedrock',
          'Could not fetch inference profiles: @error',
          ['@error' => $e->getMessage()],
          WATCHDOG_DEBUG
        );
      }

      // ---- Pass 2: Foundation models (fill gaps) ----
      $result = $client->listFoundationModels([]);
      if (!empty($result['modelSummaries'])) {
        foreach ($result['modelSummaries'] as $model) {
          // Skip non-active models.
          if (isset($model['modelLifecycle']['status'])
            && $model['modelLifecycle']['status'] !== 'ACTIVE'
          ) {
            continue;
          }

          $id = $model['modelId'];

          // Skip if an inference profile already covers this model.
          if (isset($profile_base_ids[$id])) {
            continue;
          }

          // Skip context-window / throughput variants (2+ colons).
          if (substr_count($id, ':') >= 2) {
            continue;
          }

          // Skip non-text models (image generators, video, audio, rerank).
          if ($this->isNonTextModel($id)) {
            continue;
          }

          $name = !empty($model['modelName'])
            ? $model['modelName'] . ' (' . $id . ')'
            : $id;
          $models[$id] = $name;
        }
      }

      if (!empty($models)) {
        asort($models);
        return $models;
      }
    }
    catch (\Exception $e) {
      watchdog(
        'ai_provider_aws_bedrock',
        'Failed to fetch Bedrock models: @error',
        ['@error' => $e->getMessage()],
        WATCHDOG_ERROR
      );
    }

    // Return a curated fallback list if the API call fails.
    return $this->getFallbackModels();
  }

  /**
   * Check if a model ID represents a non-text model.
   *
   * Filters out image generators, video models, audio/speech models, rerank
   * models, and other non-chat/non-embedding models that are not usable
   * through this adapter.
   *
   * @param string $model_id
   *   The model ID or inference profile ID.
   *
   * @return bool
   *   TRUE if the model should be excluded from the list.
   */
  protected function isNonTextModel(string $model_id): bool {
    // Strip region prefix for matching.
    $bare_id = preg_replace('/^(us|eu|ap|global)\./', '', $model_id);

    // Image generation models.
    if (preg_match('/\b(stable-|titan-image|nova-canvas|nova-reel)/i', $bare_id)) {
      return TRUE;
    }

    // Audio / speech models.
    if (preg_match('/\b(sonic|voxtral|pegasus|marengo)/i', $bare_id)) {
      return TRUE;
    }

    // Rerank models.
    if (strpos($bare_id, 'rerank') !== FALSE) {
      return TRUE;
    }

    // Safeguard / moderation-only models.
    if (strpos($bare_id, 'safeguard') !== FALSE) {
      return TRUE;
    }

    return FALSE;
  }

  /**
   * Get models by their capability.
   *
   * @param string $capability
   *   The capability to filter by (e.g., 'text', 'vision', 'embedding').
   *
   * @return array
   *   Array of model_id => label.
   */
  public function getModelsByCapability($capability): array {
    $models = $this->getModels();
    $filtered = [];

    foreach ($models as $id => $label) {
      $is_match = FALSE;

      switch ($capability) {
        case 'text':
          $is_match = strpos($id, 'embed') === FALSE && strpos($id, 'rerank') === FALSE;
          break;

        case 'embedding':
        case 'embeddings':
          $is_match = strpos($id, 'embed') !== FALSE;
          break;

        case 'vision':
          $is_match = (bool) preg_match(
            '/anthropic\.claude-3|anthropic\.claude-4|meta\.llama3.*vision|amazon\.nova-.*(pro|lite|sonic)|google\.gemma-3/i',
            $id
          );
          break;

        case 'image':
        case 'moderation':
        case 'tts':
        default:
          $is_match = FALSE;
          break;
      }

      if ($is_match) {
        $filtered[$id] = $label;
      }
    }

    backdrop_alter('ai_model_capabilities', $filtered, $capability, $this);

    return $filtered;
  }

  /**
   * {@inheritdoc}
   */
  public function completions(
    string $model,
    string $prompt,
    $temperature,
    $max_tokens = 512,
    bool $stream_response = FALSE
  ) {
    // Route completions through the chat method using the Converse API.
    $messages = [
      ['role' => 'user', 'content' => $prompt],
    ];
    return $this->chat(
      $model,
      $messages,
      $temperature,
      $max_tokens,
      $stream_response
    );
  }

  /**
   * {@inheritdoc}
   */
  public function chat(
    string $model,
    array $messages,
    $temperature,
    $max_tokens = 1024,
    bool $stream_response = FALSE,
    array $context_extra = []
  ) {
    try {
      // Allow other modules to alter chat messages before sending.
      if (function_exists('backdrop_alter')) {
        $context = [
          'operation' => 'chat',
          'model'     => $model,
          'provider'  => 'aws_bedrock',
        ];
        if (empty($context_extra['skip_ai_message_alter'])) {
          backdrop_alter('ai_chat_messages', $messages, $context);
        }
      }

      $bedrock_messages = $this->convertMessages($messages);
      $system_messages = $this->extractSystemMessages($messages);

      $payload = [
        'modelId'         => $model,
        'messages'        => $bedrock_messages,
        'inferenceConfig' => [
          'maxTokens'   => (int) $max_tokens ?: 1024,
          'temperature' => (float) $temperature,
        ],
      ];

      if (!empty($system_messages)) {
        $payload['system'] = $system_messages;
      }

      if ($stream_response) {
        // Also apply inference profile resolution for streaming.
        try {
          return $this->handleStreamingResponse($payload);
        }
        catch (\Exception $e) {
          $needs_profile = strpos($e->getMessage(), 'on-demand throughput isn\'t supported') !== FALSE;
          $lacks_prefix = !preg_match('/^(us|eu|ap)\./', $model);
          if ($needs_profile && $lacks_prefix) {
            $payload['modelId'] = $this->resolveInferenceProfileId($model);
            return $this->handleStreamingResponse($payload);
          }
          throw $e;
        }
      }

      $client = $this->getRuntimeClient();

      try {
        $response = $client->converse($payload);
      }
      catch (\Exception $e) {
        // If the model requires an inference profile, retry with the
        // region-prefixed inference profile ID.
        $needs_profile = strpos($e->getMessage(), 'on-demand throughput isn\'t supported') !== FALSE;
        $lacks_prefix = !preg_match('/^(us|eu|ap)\./', $model);
        if ($needs_profile && $lacks_prefix) {
          $profile_id = $this->resolveInferenceProfileId($model);
          watchdog(
            'ai_provider_aws_bedrock',
            'Model @model requires inference profile, retrying with @profile.',
            ['@model' => $model, '@profile' => $profile_id],
            WATCHDOG_INFO
          );
          $payload['modelId'] = $profile_id;
          $response = $client->converse($payload);
        }
        else {
          throw $e;
        }
      }

      // Extract text from the Converse API response.
      $text = '';
      if (isset($response['output']['message']['content'])) {
        foreach ($response['output']['message']['content'] as $block) {
          if (isset($block['text'])) {
            $text .= $block['text'];
          }
        }
      }

      return trim($text);
    }
    catch (\Exception $e) {
      watchdog(
        'ai_provider_aws_bedrock',
        'Chat error: @error',
        ['@error' => $e->getMessage()],
        WATCHDOG_ERROR
      );
      throw $e;
    }
  }

  /**
   * {@inheritdoc}
   */
  public function images(
    string $model,
    string $prompt,
    string $size,
    string $response_format,
    string $quality = 'standard',
    string $style = 'natural',
    ?string $output_format = NULL
  ) {
    // Bedrock supports image generation via Titan Image and Stable Diffusion,
    // but the shared AI interface is not well suited for these models.
    watchdog(
      'ai_provider_aws_bedrock',
      'Image generation via Bedrock is not yet supported through this adapter.',
      [],
      WATCHDOG_WARNING
    );
    throw new \RuntimeException('Image generation is not supported by the AWS Bedrock adapter.');
  }

  /**
   * {@inheritdoc}
   */
  public function textToSpeech(
    string $model,
    string $input,
    string $voice,
    string $response_format
  ) {
    watchdog(
      'ai_provider_aws_bedrock',
      'Text-to-speech is not supported by AWS Bedrock.',
      [],
      WATCHDOG_WARNING
    );
    throw new \RuntimeException('Text-to-speech is not supported by AWS Bedrock.');
  }

  /**
   * {@inheritdoc}
   */
  public function speechToText(
    string $model,
    string $file,
    string $task = 'transcribe',
    $temperature = 0.4,
    string $response_format = 'verbose_json'
  ) {
    watchdog(
      'ai_provider_aws_bedrock',
      'Speech-to-text is not supported by AWS Bedrock.',
      [],
      WATCHDOG_WARNING
    );
    throw new \RuntimeException('Speech-to-text is not supported by AWS Bedrock.');
  }

  /**
   * {@inheritdoc}
   */
  public function moderation(
    string $input,
    string $model = 'bedrock-moderation'
  ): array {
    watchdog(
      'ai_provider_aws_bedrock',
      'Moderation API is not directly supported by AWS Bedrock.',
      [],
      WATCHDOG_WARNING
    );
    throw new \RuntimeException('Moderation is not supported by AWS Bedrock.');
  }

  /**
   * {@inheritdoc}
   */
  public function embedding(
    string $input,
    string $model,
    bool $log = TRUE
  ): array {
    try {
      $client = $this->getRuntimeClient();

      // Build the payload based on the model type.
      if (strpos($model, 'amazon.titan-embed') === 0) {
        $payload = ['inputText' => $input];
      }
      elseif (strpos($model, 'cohere.embed') === 0) {
        $payload = [
          'texts'      => [$input],
          'input_type' => 'search_document',
        ];
      }
      else {
        // Generic fallback.
        $payload = ['inputText' => $input];
      }

      $invoke_params = [
        'modelId'     => $model,
        'body'        => json_encode($payload),
        'contentType' => 'application/json',
      ];

      try {
        $response = $client->invokeModel($invoke_params);
      }
      catch (\Exception $e) {
        // Retry with inference profile ID if on-demand isn't supported.
        $needs_profile = strpos($e->getMessage(), 'on-demand throughput isn\'t supported') !== FALSE;
        $lacks_prefix = !preg_match('/^(us|eu|ap)\./', $model);
        if ($needs_profile && $lacks_prefix) {
          $invoke_params['modelId'] = $this->resolveInferenceProfileId($model);
          $response = $client->invokeModel($invoke_params);
        }
        else {
          throw $e;
        }
      }

      $body = json_decode((string) $response['body'], TRUE);

      // Extract embeddings based on model type.
      if (strpos($model, 'amazon.titan-embed') === 0) {
        return $body['embedding'] ?? [];
      }
      elseif (strpos($model, 'cohere.embed') === 0) {
        return $body['embeddings'][0] ?? [];
      }

      return $body['embedding'] ?? [];
    }
    catch (\Exception $e) {
      if ($log) {
        watchdog(
          'ai_provider_aws_bedrock',
          'Embedding error: @error',
          ['@error' => $e->getMessage()],
          WATCHDOG_ERROR
        );
      }
      throw $e;
    }
  }

  /**
   * Handle streaming responses via the Converse Stream API.
   *
   * @param array $payload
   *   The Converse API payload.
   *
   * @return object
   *   A streaming response object.
   */
  protected function handleStreamingResponse(array $payload) {
    try {
      $client = $this->getRuntimeClient();
      $response = $client->converseStream($payload);

      $stream = $response->get('stream');

      return new class($stream) {

        protected $stream;

        public function __construct($stream) {
          $this->stream = $stream;
        }

        public function send() {
          foreach ($this->stream as $event) {
            if (isset($event['contentBlockDelta']['delta']['text'])) {
              echo $event['contentBlockDelta']['delta']['text'];
              @ob_flush();
              @flush();
            }
          }
        }

      };
    }
    catch (\Exception $e) {
      watchdog(
        'ai_provider_aws_bedrock',
        'Streaming error: @error',
        ['@error' => $e->getMessage()],
        WATCHDOG_ERROR
      );
      throw $e;
    }
  }

  /**
   * Convert shared chat-style messages to Bedrock Converse API format.
   *
   * @param array $messages
   *   Array of messages in the shared chat format.
   *
   * @return array
   *   Converted messages for the Bedrock Converse API.
   */
  protected function convertMessages(array $messages): array {
    $bedrock_messages = [];

    foreach ($messages as $msg) {
      $role = $msg['role'] ?? 'user';

      // System messages are handled separately via the 'system' parameter.
      if ($role === 'system') {
        continue;
      }

      // Bedrock Converse API supports 'user' and 'assistant' roles.
      $bedrock_role = ($role === 'assistant') ? 'assistant' : 'user';

      $content = $msg['content'] ?? '';

      // Content can be a string or an array of content blocks.
      if (is_string($content)) {
        $bedrock_messages[] = [
          'role'    => $bedrock_role,
          'content' => [
            ['text' => $content],
          ],
        ];
      }
      elseif (is_array($content)) {
        $blocks = [];
        foreach ($content as $block) {
          $blocks[] = $this->convertContentBlock($block, $bedrock_role);
        }
        $bedrock_messages[] = [
          'role'    => $bedrock_role,
          'content' => $blocks,
        ];
      }
    }

    return $bedrock_messages;
  }

  /**
   * Convert a single shared chat content block into Bedrock format.
   *
   * @param mixed $block
   *   The incoming content block.
   * @param string $role
   *   The resolved Bedrock role for the parent message.
   *
   * @return array
   *   A Bedrock content block.
   */
  protected function convertContentBlock($block, string $role): array {
    if (is_string($block)) {
      return ['text' => $block];
    }

    if (!is_array($block)) {
      return ['text' => json_encode($block)];
    }

    $type = $block['type'] ?? NULL;

    if ($type === 'text') {
      return ['text' => $block['text'] ?? ''];
    }

    if ($type === 'image_url') {
      if ($role !== 'user') {
        throw new \Exception('AWS Bedrock only supports image inputs on user messages.');
      }

      return [
        'image' => $this->buildImageContentBlock($block['image_url'] ?? NULL),
      ];
    }

    return ['text' => json_encode($block)];
  }

  /**
   * Build a Bedrock image content block from a shared chat image reference.
   *
   * @param mixed $image_reference
   *   The image reference from the shared chat message.
   *
   * @return array
   *   A Bedrock image block.
   */
  protected function buildImageContentBlock($image_reference): array {
    $source = '';
    if (is_array($image_reference)) {
      $source = $image_reference['url'] ?? '';
    }
    elseif (is_string($image_reference)) {
      $source = $image_reference;
    }

    if ($source === '') {
      throw new \Exception('AWS Bedrock image input requires a valid image URL or data URI.');
    }

    $image = $this->loadImageBytes($source);

    return [
      'format' => $image['format'],
      'source' => [
        'bytes' => $image['bytes'],
      ],
    ];
  }

  /**
   * Load image bytes and format for Bedrock image inputs.
   *
   * @param string $source
   *   The image source. Supports data URIs, local paths, and HTTP(S) URLs.
   *
   * @return array
   *   Array with keys 'bytes' and 'format'.
   */
  protected function loadImageBytes(string $source): array {
    if (strpos($source, 'data:image/') === 0) {
      return $this->decodeDataUriImage($source);
    }

    if (preg_match('#^https?://#i', $source)) {
      $response = backdrop_http_request($source, [
        'method' => 'GET',
        'timeout' => 15,
        'headers' => [
          'Accept' => 'image/*',
        ],
      ]);

      if (empty($response->data) || !isset($response->code) || (int) $response->code !== 200) {
        throw new \Exception('Unable to download the image URL for AWS Bedrock input.');
      }

      return [
        'bytes' => $response->data,
        'format' => $this->detectImageFormat($source, $this->extractContentType($response)),
      ];
    }

    $bytes = @file_get_contents($source);
    if ($bytes === FALSE || $bytes === '') {
      throw new \Exception('Unable to read the image source for AWS Bedrock input.');
    }

    return [
      'bytes' => $bytes,
      'format' => $this->detectImageFormat($source),
    ];
  }

  /**
   * Decode a base64 data URI into bytes and a Bedrock-compatible format.
   *
   * @param string $source
   *   The data URI source.
   *
   * @return array
   *   Array with keys 'bytes' and 'format'.
   */
  protected function decodeDataUriImage(string $source): array {
    if (!preg_match('#^data:image/([a-zA-Z0-9.+-]+);base64,(.+)$#s', $source, $matches)) {
      throw new \Exception('Invalid image data URI supplied to AWS Bedrock.');
    }

    $bytes = base64_decode($matches[2], TRUE);
    if ($bytes === FALSE) {
      throw new \Exception('Invalid base64 image data supplied to AWS Bedrock.');
    }

    return [
      'bytes' => $bytes,
      'format' => $this->normalizeImageFormat($matches[1]),
    ];
  }

  /**
   * Extract the content type header from a Backdrop HTTP response object.
   *
   * @param object $response
   *   The Backdrop HTTP response object.
   *
   * @return string
   *   The content type, if available.
   */
  protected function extractContentType($response): string {
    if (empty($response->headers) || !is_array($response->headers)) {
      return '';
    }

    foreach ($response->headers as $name => $value) {
      if (strtolower($name) !== 'content-type') {
        continue;
      }

      if (is_array($value)) {
        return (string) reset($value);
      }

      return (string) $value;
    }

    return '';
  }

  /**
   * Detect the image format Bedrock expects for an image source.
   *
   * @param string $source
   *   The image source.
   * @param string $content_type
   *   Optional content type header.
   *
   * @return string
   *   A normalized Bedrock image format.
   */
  protected function detectImageFormat(string $source, string $content_type = ''): string {
    if ($content_type !== '') {
      if (preg_match('#image/([a-zA-Z0-9.+-]+)#i', $content_type, $matches)) {
        return $this->normalizeImageFormat($matches[1]);
      }
    }

    $path = parse_url($source, PHP_URL_PATH);
    if (!is_string($path) || $path === '') {
      $path = $source;
    }

    return $this->normalizeImageFormat(pathinfo($path, PATHINFO_EXTENSION));
  }

  /**
   * Normalize image extensions to Bedrock format names.
   *
   * @param string $format
   *   The detected image format.
   *
   * @return string
   *   A normalized image format for Bedrock.
   */
  protected function normalizeImageFormat(string $format): string {
    $format = strtolower(trim($format));
    $format = preg_replace('/\+xml$/', '', $format);

    if ($format === 'jpg') {
      $format = 'jpeg';
    }

    if (!in_array($format, ['png', 'jpeg', 'gif', 'webp'], TRUE)) {
      throw new \Exception('Unsupported image format for AWS Bedrock. Use PNG, JPEG, GIF, or WEBP.');
    }

    return $format;
  }

  /**
   * Extract system messages from the message array.
   *
   * @param array $messages
   *   Array of messages in the shared chat format.
   *
   * @return array
   *   System messages formatted for the Bedrock Converse API.
   */
  protected function extractSystemMessages(array $messages): array {
    $system = [];
    foreach ($messages as $msg) {
      if (isset($msg['role']) && $msg['role'] === 'system') {
        $text = is_string($msg['content'])
          ? $msg['content']
          : json_encode(
            $msg['content']
          );
        $system[] = ['text' => trim($text)];
      }
    }
    return $system;
  }

  /**
   * Get the inference profile region prefix based on the configured region.
   *
   * AWS requires certain newer models to be invoked via inference profiles
   * rather than direct model IDs. The inference profile ID is the model ID
   * prefixed with a region group identifier (e.g., 'us.', 'eu.', 'ap.').
   *
   * @return string
   *   The region prefix (e.g., 'us.', 'eu.', 'ap.').
   */
  protected function getRegionPrefix(): string {
    if (strpos($this->region, 'us-') === 0 || strpos($this->region, 'ca-') === 0 || strpos($this->region, 'sa-') === 0) {
      return 'us.';
    }
    if (strpos($this->region, 'eu-') === 0) {
      return 'eu.';
    }
    if (strpos($this->region, 'ap-') === 0) {
      return 'ap.';
    }
    return 'us.';
  }

  /**
   * Resolve a model ID to an inference profile ID if needed.
   *
   * If the model ID already has a region prefix (e.g., 'us.anthropic...'),
   * it is returned as-is. Otherwise, the region prefix is prepended.
   *
   * @param string $model_id
   *   The original model ID.
   *
   * @return string
   *   The model ID with region prefix prepended.
   */
  protected function resolveInferenceProfileId(string $model_id): string {
    // Already has a region prefix.
    if (preg_match('/^(us|eu|ap)\./', $model_id)) {
      return $model_id;
    }
    return $this->getRegionPrefix() . $model_id;
  }

  /**
   * {@inheritdoc}
   */
  public function chatWithTools(string $model, array $messages, array $tools, $temperature, $max_tokens = 1024, string $tool_choice = 'auto', array $context_extra = []): array {
    try {
      $bedrock_messages = $this->convertMessages($messages);
      $system_messages = $this->extractSystemMessages($messages);

      $payload = [
        'modelId'         => $model,
        'messages'        => $bedrock_messages,
        'inferenceConfig' => [
          'maxTokens'   => (int) $max_tokens ?: 1024,
          'temperature' => (float) $temperature,
        ],
      ];

      if (!empty($system_messages)) {
        $payload['system'] = $system_messages;
      }

      // Convert shared tools to Bedrock toolConfig.
      $bedrock_tools = [];
      foreach ($tools as $tool) {
        if (($tool['type'] ?? '') === 'function') {
          $bedrock_tools[] = [
            'toolSpec' => [
              'name'        => $tool['function']['name'],
              'description' => $tool['function']['description'] ?? '',
              'inputSchema' => [
                'json' => $tool['function']['parameters'] ?? (object) [],
              ],
            ],
          ];
        }
      }

      if (!empty($bedrock_tools)) {
        $payload['toolConfig'] = [
          'tools' => $bedrock_tools,
        ];
        if ($tool_choice === 'required') {
          $payload['toolConfig']['toolChoice'] = ['any' => (object) []];
        }
        elseif ($tool_choice !== 'auto' && $tool_choice !== 'none') {
          // If it's a specific function name.
          $payload['toolConfig']['toolChoice'] = ['tool' => ['name' => $tool_choice]];
        }
      }

      $client = $this->getRuntimeClient();
      try {
        $response = $client->converse($payload);
      }
      catch (\Exception $e) {
        $needs_profile = strpos($e->getMessage(), 'on-demand throughput isn\'t supported') !== FALSE;
        $lacks_prefix = !preg_match('/^(us|eu|ap)\./', $model);
        if ($needs_profile && $lacks_prefix) {
          $payload['modelId'] = $this->resolveInferenceProfileId($model);
          $response = $client->converse($payload);
        }
        else {
          throw $e;
        }
      }

      $content = '';
      $tool_calls = [];
      if (isset($response['output']['message']['content'])) {
        foreach ($response['output']['message']['content'] as $block) {
          if (isset($block['text'])) {
            $content .= $block['text'];
          }
          if (isset($block['toolUse'])) {
            $tool_calls[] = [
              'id'        => $block['toolUse']['toolUseId'],
              'name'      => $block['toolUse']['name'],
              'arguments' => json_encode($block['toolUse']['input']),
            ];
          }
        }
      }

      $stop_reason = $response['stopReason'] ?? 'stop';
      $finish_reason = 'stop';
      if ($stop_reason === 'tool_use') {
        $finish_reason = 'tool_calls';
      }
      elseif ($stop_reason === 'end_turn') {
        $finish_reason = 'stop';
      }
      elseif ($stop_reason === 'max_tokens') {
        $finish_reason = 'length';
      }

      return [
        'finish_reason' => $finish_reason,
        'content'       => trim($content),
        'tool_calls'    => $tool_calls,
        'raw'           => (array) $response,
      ];
    }
    catch (\Exception $e) {
      watchdog('ai_provider_aws_bedrock', 'chatWithTools error: @error', ['@error' => $e->getMessage()], WATCHDOG_ERROR);
      throw $e;
    }
  }

  /**
   * Return a curated fallback list of popular Bedrock models.
   *
   * @return array
   *   Array of model_id => label.
   */
  protected function getFallbackModels(): array {
    return [
      'anthropic.claude-3-5-sonnet-20240620-v1:0' => 'Claude 3.5 Sonnet (anthropic.claude-3-5-sonnet-20240620-v1:0)',
      'anthropic.claude-3-haiku-20240307-v1:0'    => 'Claude 3 Haiku (anthropic.claude-3-haiku-20240307-v1:0)',
      'anthropic.claude-3-opus-20240229-v1:0'     => 'Claude 3 Opus (anthropic.claude-3-opus-20240229-v1:0)',
      'meta.llama3-8b-instruct-v1:0'              => 'Llama 3 8B Instruct (meta.llama3-8b-instruct-v1:0)',
      'meta.llama3-70b-instruct-v1:0'             => 'Llama 3 70B Instruct (meta.llama3-70b-instruct-v1:0)',
      'mistral.mistral-large-2402-v1:0'           => 'Mistral Large (mistral.mistral-large-2402-v1:0)',
      'mistral.mixtral-8x7b-instruct-v0:1'        => 'Mixtral 8x7B (mistral.mixtral-8x7b-instruct-v0:1)',
      'amazon.titan-text-express-v1'              => 'Titan Text Express (amazon.titan-text-express-v1)',
      'amazon.titan-embed-text-v1'                => 'Titan Embeddings G1 (amazon.titan-embed-text-v1)',
      'cohere.command-r-v1:0'                     => 'Command R (cohere.command-r-v1:0)',
    ];
  }

}
