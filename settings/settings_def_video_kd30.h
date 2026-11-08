/* Single-source definitions: KD-30 CRT output (Vulkan).
 * Grammar identical to settings_def_video_sync.h; row order is menu
 * display order. */

S_BOOL(video_kd30, VIDEO_KD30,
      "video_kd30",
      DEFAULT_VIDEO_KD30, SD_FLAG_NONE, 0, CMD_EVENT_NONE,
      "KD-30 CRT Output",
      "Show the game on a simulated Sony KD-30XS955 tube instead of the shader chain. Vulkan only.")
S_UINT(video_kd30_mask, VIDEO_KD30_MASK,
      "video_kd30_mask",
      DEFAULT_VIDEO_KD30_MASK, SD_FLAG_NONE, SDESC_RANGE_MINMAX, CMD_EVENT_NONE, 0, 4, 1, 0, setting_action_ok_uint, setting_get_string_representation_uint_video_kd30_mask,
      "KD-30 Phosphor Mask",
      "Phosphor layout of the simulated tube.")
S_BOOL(video_kd30_beam, VIDEO_KD30_BEAM,
      "video_kd30_beam",
      DEFAULT_VIDEO_KD30_BEAM, SD_FLAG_NONE, 0, CMD_EVENT_NONE,
      "KD-30 Beam",
      "Each display refresh shows the light the tube emits during it, as the beam sweeps down. Off shows each field's average.")
S_UINT(video_kd30_luma, VIDEO_KD30_LUMA,
      "video_kd30_luma",
      DEFAULT_VIDEO_KD30_LUMA, SD_FLAG_NONE, SDESC_RANGE_MINMAX, CMD_EVENT_NONE, 80, 400, 10, 0, setting_action_ok_uint, setting_get_string_representation_uint_video_kd30_luma,
      "KD-30 Luma",
      "Luminance of a full white field on the simulated tube.")
S_UINT(video_kd30_input, VIDEO_KD30_INPUT,
      "video_kd30_input",
      DEFAULT_VIDEO_KD30_INPUT, SD_FLAG_NONE, SDESC_RANGE_MINMAX, CMD_EVENT_NONE, 0, 2, 1, 0, setting_action_ok_uint, setting_get_string_representation_uint_video_kd30_input,
      "KD-30 Video Input",
      "Signal path from the console to the tube.")
