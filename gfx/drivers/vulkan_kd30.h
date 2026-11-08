/*  RetroArch - A frontend for libretro.
 *
 *  RetroArch is free software: you can redistribute it and/or modify it under the terms
 *  of the GNU General Public License as published by the Free Software Found-
 *  ation, either version 3 of the License, or (at your option) any later version.
 *
 *  RetroArch is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 *  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 *  PURPOSE.  See the GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along with RetroArch.
 *  If not, see <http://www.gnu.org/licenses/>.
 */

/* KD-30XS955 CRT output: a Sony WEGA tube simulated in two compute
 * passes, drawn in place of the shader chain. Included by vulkan.c. */

#include <ctype.h>

#include "../../runloop.h"

#define KD30_SCAN_W     2560
#define KD30_DEAD_MAX   8

struct kd30_line_push
{
   uint32_t raw_w, raw_h, scan_w, scan_h;
   uint32_t frame_count, scan_format, signal_standard, input_mode;
   float vm_strength, horiz_smear, comb_strength;
   float sc_cyc_per_px, sc_line_step, sc_frame_step, sc_phase0;
};

struct kd30_scan_push
{
   uint32_t out_w, out_h, scan_w, scan_h, raw_w, raw_h;
   uint32_t frame_count, scan_format, signal_standard, input_mode;
   uint32_t osd_mode, osd_selected_row, perf_overlay_on;
   uint32_t mask_layout, aspect_mode;
   float sfp_pitch_mm, hv_breathing, damper_str, brightness_gain;
   float split_wipe, beam_sigma_scale;
   float vm_strength, horiz_smear, comb_strength;
   float perf_fps, perf_frame_ms, perf_gpu_ms;
   float zoom_factor, zoom_center_x, zoom_center_y;
   float field_ms, window_ms, cur_start_ms, output_mode;
   float crt_white_nits, display_peak_nits, sdr_ref_nits, panel_tau_ms;
};

struct kd30_image
{
   VkImage image;
   VkImageView view;
   VkDeviceMemory memory;
   unsigned w, h;
   bool fresh;
};

struct kd30_dead
{
   struct kd30_image img;
   unsigned frames;
};

struct kd30_state
{
   VkDescriptorSetLayout line_dsl, scan_dsl;
   VkPipelineLayout line_layout, scan_layout;
   VkPipeline line_pipe, scan_pipe;
   VkDescriptorPool pool;
   VkDescriptorSet line_set[VULKAN_MAX_SWAPCHAIN_IMAGES];
   VkDescriptorSet scan_set[VULKAN_MAX_SWAPCHAIN_IMAGES];
   VkSampler sampler;
   struct kd30_image scan[2];
   struct kd30_image accum;
   struct kd30_image out;
   struct kd30_dead dead[KD30_DEAD_MAX];
   struct kd30_line_push line;
   struct kd30_line_push line_last;
   struct kd30_scan_push sp;
   unsigned cur;
   bool line_valid;
};

static void kd30_image_free(VkDevice dev, struct kd30_image *img)
{
   if (img->view)
      vkDestroyImageView(dev, img->view, NULL);
   if (img->image)
      vkDestroyImage(dev, img->image, NULL);
   if (img->memory)
      vkFreeMemory(dev, img->memory, NULL);
   memset(img, 0, sizeof(*img));
}

/* An image a frame in flight may still use is freed once every
 * swapchain slot has come round again. */
static void kd30_image_retire(vk_t *vk, struct kd30_state *k,
      struct kd30_image *img)
{
   unsigned i;
   unsigned slot = 0;
   if (!img->image)
      return;
   for (i = 0; i < KD30_DEAD_MAX; i++)
   {
      if (!k->dead[i].img.image)
      {
         slot = i;
         break;
      }
      if (k->dead[i].frames < k->dead[slot].frames)
         slot = i;
   }
   /* All taken only if sizes change every frame: the oldest has waited. */
   kd30_image_free(vk->context->device, &k->dead[slot].img);
   k->dead[slot].img    = *img;
   k->dead[slot].frames = vk->context->num_swapchain_images + 1;
   memset(img, 0, sizeof(*img));
}

static void kd30_dead_tick(vk_t *vk, struct kd30_state *k)
{
   unsigned i;
   for (i = 0; i < KD30_DEAD_MAX; i++)
   {
      if (k->dead[i].img.image && --k->dead[i].frames == 0)
         kd30_image_free(vk->context->device, &k->dead[i].img);
   }
}

static bool kd30_image_create(vk_t *vk, struct kd30_image *img,
      unsigned w, unsigned h)
{
   VkImageCreateInfo info;
   VkImageViewCreateInfo view;
   VkMemoryAllocateInfo alloc;
   VkMemoryRequirements req;
   VkDevice dev = vk->context->device;

   memset(img, 0, sizeof(*img));
   memset(&info, 0, sizeof(info));
   info.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
   info.imageType     = VK_IMAGE_TYPE_2D;
   info.format        = VK_FORMAT_R16G16B16A16_SFLOAT;
   info.extent.width  = w;
   info.extent.height = h;
   info.extent.depth  = 1;
   info.mipLevels     = 1;
   info.arrayLayers   = 1;
   info.samples       = VK_SAMPLE_COUNT_1_BIT;
   info.tiling        = VK_IMAGE_TILING_OPTIMAL;
   info.usage         = VK_IMAGE_USAGE_STORAGE_BIT
      | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
   info.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
   info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
   if (vkCreateImage(dev, &info, NULL, &img->image) != VK_SUCCESS)
      return false;

   vkGetImageMemoryRequirements(dev, img->image, &req);
   alloc.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
   alloc.pNext           = NULL;
   alloc.allocationSize  = req.size;
   alloc.memoryTypeIndex = vulkan_find_memory_type(
         &vk->context->memory_properties, req.memoryTypeBits,
         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
   if (     vkAllocateMemory(dev, &alloc, NULL, &img->memory) != VK_SUCCESS
         || vkBindImageMemory(dev, img->image, img->memory, 0) != VK_SUCCESS)
   {
      kd30_image_free(dev, img);
      return false;
   }

   memset(&view, 0, sizeof(view));
   view.sType                       = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
   view.image                       = img->image;
   view.viewType                    = VK_IMAGE_VIEW_TYPE_2D;
   view.format                      = info.format;
   view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
   view.subresourceRange.levelCount = 1;
   view.subresourceRange.layerCount = 1;
   if (vkCreateImageView(dev, &view, NULL, &img->view) != VK_SUCCESS)
   {
      kd30_image_free(dev, img);
      return false;
   }
   img->w     = w;
   img->h     = h;
   img->fresh = true;
   return true;
}

static bool kd30_image_ensure(vk_t *vk, struct kd30_state *k,
      struct kd30_image *img, unsigned w, unsigned h)
{
   if (img->image && img->w == w && img->h == h)
      return true;
   kd30_image_retire(vk, k, img);
   return kd30_image_create(vk, img, w, h);
}

static VkPipeline kd30_pipeline(vk_t *vk, VkPipelineLayout layout,
      const uint32_t *code, size_t size)
{
   VkShaderModuleCreateInfo mod;
   VkComputePipelineCreateInfo info;
   VkPipeline pipe = VK_NULL_HANDLE;
   VkShaderModule shader;

   mod.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
   mod.pNext    = NULL;
   mod.flags    = 0;
   mod.codeSize = size;
   mod.pCode    = code;
   if (vkCreateShaderModule(vk->context->device, &mod, NULL, &shader)
         != VK_SUCCESS)
      return VK_NULL_HANDLE;

   memset(&info, 0, sizeof(info));
   info.sType        = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
   info.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
   info.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
   info.stage.module = shader;
   info.stage.pName  = "main";
   info.layout       = layout;
   vkCreateComputePipelines(vk->context->device, vk->pipelines.cache,
         1, &info, NULL, &pipe);
   vkDestroyShaderModule(vk->context->device, shader, NULL);
   return pipe;
}

static bool kd30_layout(VkDevice dev, const VkDescriptorType *types,
      unsigned count, uint32_t push_size,
      VkDescriptorSetLayout *dsl, VkPipelineLayout *layout)
{
   unsigned i;
   VkDescriptorSetLayoutBinding b[5];
   VkDescriptorSetLayoutCreateInfo info;
   VkPipelineLayoutCreateInfo pinfo;
   VkPushConstantRange range;

   for (i = 0; i < count; i++)
   {
      b[i].binding            = i;
      b[i].descriptorType     = types[i];
      b[i].descriptorCount    = 1;
      b[i].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
      b[i].pImmutableSamplers = NULL;
   }
   info.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
   info.pNext        = NULL;
   info.flags        = 0;
   info.bindingCount = count;
   info.pBindings    = b;
   if (vkCreateDescriptorSetLayout(dev, &info, NULL, dsl) != VK_SUCCESS)
      return false;

   range.stageFlags             = VK_SHADER_STAGE_COMPUTE_BIT;
   range.offset                 = 0;
   range.size                   = push_size;
   pinfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
   pinfo.pNext                  = NULL;
   pinfo.flags                  = 0;
   pinfo.setLayoutCount         = 1;
   pinfo.pSetLayouts            = dsl;
   pinfo.pushConstantRangeCount = 1;
   pinfo.pPushConstantRanges    = &range;
   return vkCreatePipelineLayout(dev, &pinfo, NULL, layout) == VK_SUCCESS;
}

static void kd30_free(vk_t *vk)
{
   unsigned i;
   struct kd30_state *k = vk->kd30;
   VkDevice dev;
   if (!k)
      return;
   dev = vk->context->device;
   for (i = 0; i < KD30_DEAD_MAX; i++)
      kd30_image_free(dev, &k->dead[i].img);
   kd30_image_free(dev, &k->scan[0]);
   kd30_image_free(dev, &k->scan[1]);
   kd30_image_free(dev, &k->accum);
   kd30_image_free(dev, &k->out);
   if (k->pool)
      vkDestroyDescriptorPool(dev, k->pool, NULL);
   if (k->line_pipe)
      vkDestroyPipeline(dev, k->line_pipe, NULL);
   if (k->scan_pipe)
      vkDestroyPipeline(dev, k->scan_pipe, NULL);
   if (k->line_layout)
      vkDestroyPipelineLayout(dev, k->line_layout, NULL);
   if (k->scan_layout)
      vkDestroyPipelineLayout(dev, k->scan_layout, NULL);
   if (k->line_dsl)
      vkDestroyDescriptorSetLayout(dev, k->line_dsl, NULL);
   if (k->scan_dsl)
      vkDestroyDescriptorSetLayout(dev, k->scan_dsl, NULL);
   if (k->sampler)
      vkDestroySampler(dev, k->sampler, NULL);
   free(k);
   vk->kd30 = NULL;
}

static struct kd30_state *kd30_init(vk_t *vk)
{
   static const uint32_t line_comp[] =
#include "vulkan_shaders/kd30_line.comp.inc"
   ;
   static const uint32_t scan_comp[] =
#include "vulkan_shaders/kd30_scanout.comp.inc"
   ;
   static const VkDescriptorType line_types[2] = {
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      VK_DESCRIPTOR_TYPE_STORAGE_IMAGE };
   static const VkDescriptorType scan_types[5] = {
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER };
   unsigned i;
   VkPhysicalDeviceProperties props;
   VkSamplerCreateInfo sinfo;
   VkDescriptorPoolSize sizes[2];
   VkDescriptorPoolCreateInfo pinfo;
   VkDescriptorSetAllocateInfo ainfo;
   VkDevice dev = vk->context->device;
   struct kd30_state *k;

   vkGetPhysicalDeviceProperties(vk->context->gpu, &props);
   if (props.limits.maxPushConstantsSize < sizeof(struct kd30_scan_push))
   {
      RARCH_ERR("[KD-30] GPU allows %u bytes of push constants, %u needed.\n",
            props.limits.maxPushConstantsSize,
            (unsigned)sizeof(struct kd30_scan_push));
      return NULL;
   }

   k = (struct kd30_state*)calloc(1, sizeof(*k));
   if (!k)
      return NULL;
   vk->kd30 = k;

   memset(&sinfo, 0, sizeof(sinfo));
   sinfo.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
   sinfo.magFilter    = VK_FILTER_LINEAR;
   sinfo.minFilter    = VK_FILTER_LINEAR;
   sinfo.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
   sinfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
   sinfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
   sinfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
   sinfo.maxLod       = 0.0f;
   sinfo.borderColor  = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;

   sizes[0].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
   sizes[0].descriptorCount = 4 * VULKAN_MAX_SWAPCHAIN_IMAGES;
   sizes[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
   sizes[1].descriptorCount = 3 * VULKAN_MAX_SWAPCHAIN_IMAGES;
   pinfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
   pinfo.pNext         = NULL;
   pinfo.flags         = 0;
   pinfo.maxSets       = 2 * VULKAN_MAX_SWAPCHAIN_IMAGES;
   pinfo.poolSizeCount = 2;
   pinfo.pPoolSizes    = sizes;

   if (     vkCreateSampler(dev, &sinfo, NULL, &k->sampler) != VK_SUCCESS
         || !kd30_layout(dev, line_types, 2, sizeof(struct kd30_line_push),
            &k->line_dsl, &k->line_layout)
         || !kd30_layout(dev, scan_types, 5, sizeof(struct kd30_scan_push),
            &k->scan_dsl, &k->scan_layout)
         || vkCreateDescriptorPool(dev, &pinfo, NULL, &k->pool) != VK_SUCCESS)
   {
      kd30_free(vk);
      return NULL;
   }

   ainfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
   ainfo.pNext              = NULL;
   ainfo.descriptorPool     = k->pool;
   ainfo.descriptorSetCount = 1;
   for (i = 0; i < VULKAN_MAX_SWAPCHAIN_IMAGES; i++)
   {
      ainfo.pSetLayouts = &k->line_dsl;
      vkAllocateDescriptorSets(dev, &ainfo, &k->line_set[i]);
      ainfo.pSetLayouts = &k->scan_dsl;
      vkAllocateDescriptorSets(dev, &ainfo, &k->scan_set[i]);
   }

   k->line_pipe = kd30_pipeline(vk, k->line_layout,
         line_comp, sizeof(line_comp));
   k->scan_pipe = kd30_pipeline(vk, k->scan_layout,
         scan_comp, sizeof(scan_comp));
   if (!k->line_pipe || !k->scan_pipe)
   {
      kd30_free(vk);
      return NULL;
   }

   k->line.vm_strength   = 0.75f;
   k->line.horiz_smear   = 1.0f;
   k->line.comb_strength = 1.0f;

   k->sp.sfp_pitch_mm      = 0.45f;
   k->sp.hv_breathing      = 0.005f;
   k->sp.damper_str        = 0.14f;
   k->sp.brightness_gain   = 1.0f;
   k->sp.beam_sigma_scale  = 1.0f;
   k->sp.aspect_mode       = 2;
   k->sp.crt_white_nits    = 160.0f;
   k->sp.display_peak_nits = 1000.0f;
   k->sp.sdr_ref_nits      = 100.0f;
   RARCH_LOG("[KD-30] CRT output ready.\n");
   return k;
}

static bool kd30_has(const char *s, const char *key)
{
   return strstr(s, key) != NULL;
}

/* Colour subcarrier cycles per source pixel, and its per-line and
 * per-frame phase advance, for the console the core emulates. */
static void kd30_subcarrier(struct kd30_line_push *p, const char *name)
{
   char lib[64];
   size_t i;
   bool pal   = p->signal_standard == 1;
   float w    = (float)(p->raw_w ? p->raw_w : 1);
   float cpp;
   float line  = 0.0f;
   float frame = 0.0f;

   for (i = 0; name && name[i] && i < sizeof(lib) - 1; i++)
      lib[i] = (char)tolower((unsigned char)name[i]);
   lib[i] = '\0';

   if (     kd30_has(lib, "genesis") || kd30_has(lib, "picodrive")
         || kd30_has(lib, "blastem") || kd30_has(lib, "mega"))
      cpp = pal ? ((w > 300.0f) ? (2.0f / 3.0f) : (5.0f / 6.0f))
                : ((w > 300.0f) ? (8.0f / 15.0f) : (10.0f / 15.0f));
   else if (kd30_has(lib, "snes") || kd30_has(lib, "mesen-s"))
   {
      cpp = pal ? (5.0f / 6.0f) : (2.0f / 3.0f);
      if (w > 300.0f)
         cpp *= 0.5f;
      line  = pal ? 0.0f : (1.0f / 3.0f);
      frame = line;
   }
   else if (kd30_has(lib, "nes") || kd30_has(lib, "fceu")
         || kd30_has(lib, "mesen"))
   {
      cpp   = 2.0f / 3.0f;
      line  = 1.0f / 3.0f;
      frame = 1.0f / 3.0f;
   }
   else if (kd30_has(lib, "pcsx") || kd30_has(lib, "swanstation")
         || kd30_has(lib, "psx") || kd30_has(lib, "duckstation"))
   {
      float div = (w > 600.0f) ? 4.0f : (w > 500.0f) ? 5.0f
                : (w > 350.0f) ? 7.0f : (w > 300.0f) ? 8.0f : 10.0f;
      cpp   = div / 15.0f;
      line  = 0.5333f;
      frame = 0.5f;
   }
   else
   {
      cpp   = (pal ? 4.43361875f : 3.579545f) * 52.6f / w;
      line  = pal ? 0.75f : 0.5f;
      frame = pal ? 0.0f : 0.5f;
   }
   p->sc_cyc_per_px = cpp;
   p->sc_line_step  = line;
   p->sc_frame_step = frame;
}

static void kd30_barrier(VkCommandBuffer cmd,
      VkPipelineStageFlags src, VkAccessFlags src_access,
      VkPipelineStageFlags dst, VkAccessFlags dst_access)
{
   VkMemoryBarrier b;
   b.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
   b.pNext         = NULL;
   b.srcAccessMask = src_access;
   b.dstAccessMask = dst_access;
   vkCmdPipelineBarrier(cmd, src, dst, 0, 1, &b, 0, NULL, 0, NULL);
}

static void kd30_prepare_image(VkCommandBuffer cmd, struct kd30_image *img)
{
   VkClearColorValue black;
   VkImageSubresourceRange range;
   if (!img->fresh)
      return;
   memset(&black, 0, sizeof(black));
   range.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
   range.baseMipLevel   = 0;
   range.levelCount     = 1;
   range.baseArrayLayer = 0;
   range.layerCount     = 1;
   VULKAN_IMAGE_LAYOUT_TRANSITION(cmd, img->image,
         VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
         0, VK_ACCESS_TRANSFER_WRITE_BIT,
         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
   vkCmdClearColorImage(cmd, img->image, VK_IMAGE_LAYOUT_GENERAL,
         &black, 1, &range);
   img->fresh = false;
}

static void kd30_write_image(VkWriteDescriptorSet *w,
      VkDescriptorImageInfo *info, VkDescriptorSet set, unsigned binding,
      VkDescriptorType type, VkSampler sampler, VkImageView view,
      VkImageLayout layout)
{
   info->sampler       = sampler;
   info->imageView     = view;
   info->imageLayout   = layout;
   w->sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
   w->pNext            = NULL;
   w->dstSet           = set;
   w->dstBinding       = binding;
   w->dstArrayElement  = 0;
   w->descriptorCount  = 1;
   w->descriptorType   = type;
   w->pImageInfo       = info;
   w->pBufferInfo      = NULL;
   w->pTexelBufferView = NULL;
}

/* Runs the tube for this refresh. A new core frame is a new field:
 * the scanline pass runs only then, and the previous field stays
 * on the phosphor for the scanout pass. */
static bool kd30_render(vk_t *vk, VkCommandBuffer cmd,
      const struct vulkan_filter_chain_texture *input, bool new_field,
      unsigned out_w, unsigned out_h, unsigned output_mode,
      float peak_nits, unsigned mask, unsigned input_mode)
{
   VkWriteDescriptorSet w[7];
   VkDescriptorImageInfo info[7];
   struct kd30_state *k = vk->kd30;
   unsigned slot        = vk->context->current_frame_index;
   unsigned raw_w       = VIDEO_SCALE_W(input->dims);
   unsigned raw_h       = VIDEO_SCALE_H(input->dims);
   double fps           = video_state_get_ptr()->av_info.timing.fps;
   unsigned scan_h;
   struct kd30_image *scan, *prev;

   if (!k)
   {
      if (vk->kd30_failed)
         return false;
      k = kd30_init(vk);
      if (!k)
      {
         vk->kd30_failed = true;
         return false;
      }
   }
   if (!input->view || !raw_w || !raw_h || !out_w || !out_h)
      return false;
   if (fps < 1.0)
      fps = 60.0;

   kd30_dead_tick(vk, k);

   scan_h = raw_h < 64 ? 64 : raw_h > 1152 ? 1152 : raw_h;
   if (     !kd30_image_ensure(vk, k, &k->scan[0], KD30_SCAN_W, scan_h)
         || !kd30_image_ensure(vk, k, &k->scan[1], KD30_SCAN_W, scan_h)
         || !kd30_image_ensure(vk, k, &k->accum, out_w, out_h)
         || !kd30_image_ensure(vk, k, &k->out, out_w, out_h))
      return false;

   /* The last frame's draw read out; the core's frame was written
    * by a pass or a transfer before this one. */
   kd30_barrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
         VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,
         VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
         VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT
         | VK_ACCESS_SHADER_WRITE_BIT);
   if (k->scan[0].fresh || k->scan[1].fresh || k->accum.fresh || k->out.fresh)
   {
      kd30_prepare_image(cmd, &k->scan[0]);
      kd30_prepare_image(cmd, &k->scan[1]);
      kd30_prepare_image(cmd, &k->accum);
      kd30_prepare_image(cmd, &k->out);
      kd30_barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
      k->line_valid = false;
   }

   if (new_field)
   {
      k->cur       ^= 1;
      k->line_valid = false;
   }
   scan = &k->scan[k->cur];
   prev = &k->scan[k->cur ^ 1];

   k->line.raw_w           = raw_w;
   k->line.raw_h           = raw_h;
   k->line.scan_w          = KD30_SCAN_W;
   k->line.scan_h          = scan_h;
   k->line.scan_format     = raw_h >= 400 ? 1 : 0;
   k->line.signal_standard = fps < 55.0 ? 1 : 0;
   k->line.input_mode      = input_mode;
   kd30_subcarrier(&k->line,
         runloop_state_get_ptr()->system.info.library_name);

   kd30_write_image(&w[0], &info[0], k->line_set[slot], 0,
         VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, k->sampler,
         input->view, input->layout);
   kd30_write_image(&w[1], &info[1], k->line_set[slot], 1,
         VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_NULL_HANDLE,
         scan->view, VK_IMAGE_LAYOUT_GENERAL);
   kd30_write_image(&w[2], &info[2], k->scan_set[slot], 0,
         VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, k->sampler,
         scan->view, VK_IMAGE_LAYOUT_GENERAL);
   kd30_write_image(&w[3], &info[3], k->scan_set[slot], 1,
         VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_NULL_HANDLE,
         k->out.view, VK_IMAGE_LAYOUT_GENERAL);
   kd30_write_image(&w[4], &info[4], k->scan_set[slot], 2,
         VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, k->sampler,
         input->view, input->layout);
   kd30_write_image(&w[5], &info[5], k->scan_set[slot], 3,
         VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_NULL_HANDLE,
         k->accum.view, VK_IMAGE_LAYOUT_GENERAL);
   kd30_write_image(&w[6], &info[6], k->scan_set[slot], 4,
         VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, k->sampler,
         prev->view, VK_IMAGE_LAYOUT_GENERAL);
   vkUpdateDescriptorSets(vk->context->device, 7, w, 0, NULL);

   if (     !k->line_valid
         || memcmp(&k->line, &k->line_last, sizeof(k->line)) != 0)
   {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k->line_pipe);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            k->line_layout, 0, 1, &k->line_set[slot], 0, NULL);
      vkCmdPushConstants(cmd, k->line_layout, VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(k->line), &k->line);
      vkCmdDispatch(cmd, 1, scan_h, 1);
      kd30_barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
      k->line_last  = k->line;
      k->line_valid = true;
   }

   k->sp.out_w           = out_w;
   k->sp.out_h           = out_h;
   k->sp.scan_w          = KD30_SCAN_W;
   k->sp.scan_h          = scan_h;
   k->sp.raw_w           = raw_w;
   k->sp.raw_h           = raw_h;
   k->sp.frame_count     = k->line.frame_count;
   k->sp.scan_format     = k->line.scan_format;
   k->sp.signal_standard = k->line.signal_standard;
   k->sp.input_mode      = k->line.input_mode;
   k->sp.vm_strength     = k->line.vm_strength;
   k->sp.horiz_smear     = k->line.horiz_smear;
   k->sp.comb_strength   = k->line.comb_strength;
   k->sp.mask_layout     = mask;
   k->sp.field_ms        = (float)(1000.0 / fps);
   k->sp.output_mode     = (float)output_mode;
   k->sp.sdr_ref_nits    = output_mode == 1 ? 80.0f : 100.0f;
   if (peak_nits > 0.0f)
      k->sp.display_peak_nits = peak_nits;

   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k->scan_pipe);
   vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
         k->scan_layout, 0, 1, &k->scan_set[slot], 0, NULL);
   vkCmdPushConstants(cmd, k->scan_layout, VK_SHADER_STAGE_COMPUTE_BIT,
         0, sizeof(k->sp), &k->sp);
   vkCmdDispatch(cmd, (out_w + 15) / 16, (out_h + 15) / 16, 1);
   kd30_barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
         VK_ACCESS_SHADER_WRITE_BIT,
         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);

   if (new_field)
      k->line.frame_count++;
   return true;
}

/* Draws the tube where the shader chain's last pass would go. */
static void kd30_draw(vk_t *vk)
{
   struct
   {
      math_matrix_4x4 mvp;
      float clamp[4];
   } ubo;
   struct vk_texture tex;
   struct vk_draw_triangles call;
   struct vk_buffer_range range;
   struct kd30_state *k = vk->kd30;
   VkViewport saved_vp  = vk->vk_vp;
   uint32_t saved_flag  = vk->flags & VK_FLAG_TRACKER_USE_SCISSOR;
   float tw             = (float)k->out.w;
   float th             = (float)k->out.h;

   memset(&tex, 0, sizeof(tex));
   tex.view   = k->out.view;
   tex.layout = VK_IMAGE_LAYOUT_GENERAL;
   tex.dims   = VIDEO_SCALE_PACK(k->out.w, k->out.h);
   tex.format = VK_FORMAT_R16G16B16A16_SFLOAT;

   ubo.mvp      = vk->mvp;
   ubo.clamp[0] = 0.5f / tw;
   ubo.clamp[1] = 0.5f / th;
   ubo.clamp[2] = 1.0f - 0.5f / tw;
   ubo.clamp[3] = 1.0f - 0.5f / th;

   vk->vk_vp          = vk->video_vp;
   vk->flags         &= ~VK_FLAG_TRACKER_USE_SCISSOR;
   vk->tracker.dirty |= VULKAN_DIRTY_DYNAMIC_BIT;

   call.pipeline     = vk->pipelines.stock_rect;
   call.texture      = &tex;
   call.sampler      = vk->samplers.nearest;
   call.uniform      = &ubo;
   call.uniform_size = sizeof(ubo);
   call.vbo          = &range;
   call.vertices     = 4;
   if (     vulkan_draw_bind(vk, &call)
         && vulkan_buffer_chain_alloc(vk->context, &vk->chain->vbo,
            4 * sizeof(struct vk_vertex), &range))
   {
      struct vk_vertex *pv = (struct vk_vertex*)range.data;
      VULKAN_WRITE_QUAD_VBO(pv, 0.0f, 0.0f, 1.0f, 1.0f,
            0.0f, 0.0f, 1.0f, 1.0f, RGBA16_WHITE);
      vkCmdBindVertexBuffers(vk->cmd, 0, 1, &range.buffer, &range.offset);
      if (     vk->quad_ibo.buffer != VK_NULL_HANDLE
            && vk->quad_ibo.num_quads >= 1)
      {
         vkCmdBindIndexBuffer(vk->cmd, vk->quad_ibo.buffer,
               0, VK_INDEX_TYPE_UINT16);
         vkCmdDrawIndexed(vk->cmd, 6, 1, 0, 0, 0);
      }
      else
         vkCmdDraw(vk->cmd, 4, 1, 0, 0);
   }

   vk->vk_vp          = saved_vp;
   vk->flags          = (vk->flags & ~VK_FLAG_TRACKER_USE_SCISSOR) | saved_flag;
   vk->tracker.dirty |= VULKAN_DIRTY_DYNAMIC_BIT;
}
