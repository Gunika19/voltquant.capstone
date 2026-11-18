// ============================================================================
// voltquant_frame.h  -  Frame packing, AES-CMAC-128 tag generation, and decoding
//
// 16-byte Frame Layout:
//   Bytes [0..3]   : uint32_t seq (big endian)
//   Byte  [4]      : uint8_t  cell_id (0-255)
//   Byte  [5]      : uint8_t  alert_class (0=SAFE, 1=WARNING, 2=DEGRADED)
//   Byte  [6]      : uint8_t  soh (0-255 scaled from 0-100%)
//   Bytes [7..10]  : uint32_t timestamp_ms (big endian)
//   Bytes [11..15] : uint8_t  reserved[5] (0x00)
//
// 34-byte Transaction Layout:
//   Bytes [0..15]  : 16-byte Frame
//   Bytes [16..31] : 16-byte AES-CMAC Tag
//   Bytes [32..33] : 2-byte Padding / Status Echo (0x00)
// ============================================================================

#ifndef VOLTQUANT_FRAME_H
#define VOLTQUANT_FRAME_H

#include <stdint.h>
#include <string.h>
#include "mbedtls/aes.h"
#include "mbedtls/cipher.h"

#define VQ_TXN_LEN          34
#define VQ_FRAME_LEN        16
#define VQ_TAG_LEN          16

// Alert Classes
#define VQ_CLASS_SAFE       0
#define VQ_CLASS_WARNING    1
#define VQ_CLASS_DEGRADED   2

// Rejection Reason Codes
#define VQ_REJ_NONE         0
#define VQ_REJ_REPLAY       1
#define VQ_REJ_BAD_TAG      2
#define VQ_REJ_RESERVED     3
#define VQ_REJ_BAD_CLASS    4

// Status byte unpacking macros (returned by FPGA on MISO byte 33)
// Status byte bit allocation:
//   Bit 7: Valid flag (1 = verdict valid)
//   Bits [5..4]: Trusted alert class from FPGA
//   Bits [3..1]: Rejection reason code
//   Bit 0: Verdict (1 = ACCEPTED, 0 = REJECTED)
#define VQ_ST_VALID(st)     (((st) & 0x80) != 0 || ((st) != 0x00 && (st) != 0xFF))
#define VQ_ST_ACCEPTED(st)  (((st) & 0x01) != 0)
#define VQ_ST_CODE(st)      (((st) >> 1) & 0x07)
#define VQ_ST_CLASS(st)     (((st) >> 4) & 0x03)

// Convert float percentage (0.0 - 100.0%) to 8-bit integer (0 - 255)
static inline uint8_t vq_soh_from_percent(float p) {
    if (p <= 0.0f) return 0;
    if (p >= 100.0f) return 255;
    return (uint8_t)(p * 2.55f + 0.5f);
}

// Convert 8-bit SOH to float percentage
static inline float vq_percent_from_soh(uint8_t soh) {
    return (float)soh / 2.55f;
}

// Pack 16-byte telemetry frame
static inline void vq_pack_frame(
    uint8_t frame[VQ_FRAME_LEN],
    uint32_t seq,
    uint8_t cell,
    uint8_t cls,
    uint8_t soh,
    uint32_t ts
) {
    memset(frame, 0, VQ_FRAME_LEN);
    frame[0] = (uint8_t)((seq >> 24) & 0xFF);
    frame[1] = (uint8_t)((seq >> 16) & 0xFF);
    frame[2] = (uint8_t)((seq >> 8)  & 0xFF);
    frame[3] = (uint8_t)(seq & 0xFF);
    frame[4] = cell;
    frame[5] = cls;
    frame[6] = soh;
    frame[7] = (uint8_t)((ts >> 24) & 0xFF);
    frame[8] = (uint8_t)((ts >> 16) & 0xFF);
    frame[9] = (uint8_t)((ts >> 8)  & 0xFF);
    frame[10] = (uint8_t)(ts & 0xFF);
}

// Compute 16-byte AES-CMAC tag using mbedtls
static inline int vq_cmac16(const uint8_t key[16], const uint8_t frame[16], uint8_t tag[16]) {
    const mbedtls_cipher_info_t *cipher_info = mbedtls_cipher_info_from_type(MBEDTLS_CIPHER_AES_128_ECB);
    if (!cipher_info) return -1;
    return mbedtls_cipher_cmac(cipher_info, key, 128, frame, 16, tag);
}

// Build 34-byte SPI transaction buffer: [16 bytes frame | 16 bytes tag | 2 bytes zero/status]
static inline void vq_build_txn(
    uint8_t txn[VQ_TXN_LEN],
    const uint8_t frame[VQ_FRAME_LEN],
    const uint8_t tag[VQ_TAG_LEN]
) {
    memcpy(txn, frame, VQ_FRAME_LEN);
    memcpy(txn + VQ_FRAME_LEN, tag, VQ_TAG_LEN);
    txn[32] = 0x00;
    txn[33] = 0x00;
}

// Tamper one bit in the frame to demonstrate hardware bad-tag rejection
static inline void vq_tamper_txn(uint8_t txn[VQ_TXN_LEN]) {
    txn[5] ^= 0x01; // flip class or payload bit
}

#endif // VOLTQUANT_FRAME_H
