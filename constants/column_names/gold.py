"""Gold column names constants."""

# Date
DATE = "starting_week"

# Features
DIGITAL_TV_BVOD = "digital_tv_bvod"
INFLUENCER_MANAGEMENT = "influencer_management_influencer_management"
ONLINE_MULTIFORMAT_AMAZON = "online_multiformat_ads_transaction_amazon_retail"
ONLINE_MULTIFORMAT_TESCO = "online_multiformat_ads_transaction_tesco"
ONLINE_VIDEO_GOOGLE = "online_video_content_platforms_google_video"
ONLINE_VIDEO_YOUTUBE = "online_video_content_platforms_youtube"
PAID_SEARCH_AWARENESS_GOOGLE = "paid_search_awarenessconsideration_google"
PAID_SEARCH_TRANSACTION_AMAZON = "paid_search_transaction_amazon"
PAID_SEARCH_TRANSACTION_CITRUS = "paid_search_transaction_citrus"
PAID_SEARCH_TRANSACTION_CRITEO = "paid_search_transaction_criteo"
SOCIAL_MEDIA_AWARENESS_META = "social_media_awarenessconsideration_meta"
SOCIAL_MEDIA_AWARENESS_PINTEREST = "social_media_awarenessconsideration_pinterest"
SOCIAL_MEDIA_AWARENESS_TIKTOK = "social_media_awarenessconsideration_tik_tok"
SOCIAL_MEDIA_TRANSACTION_META = "social_media_transaction_meta"
SOCIAL_MEDIA_TRANSACTION_META_COLLAB = "social_media_transaction_meta_collab_ads"
SOCIAL_MEDIA_TRANSACTION_TESCO = "social_media_transaction_tesco"
SOCIAL_MEDIA_TRANSACTION_THG = "social_media_transaction_the_hut_group"
SOCIAL_MEDIA_TRANSACTION_TIKTOK = "social_media_transaction_tik_tok"
TESTERS_AND_MERCHANDISING = "testers_and_merchandising_testers_and_merchandising"
TRADITIONAL_TV_LINEAR = "traditional_tv_linear"

FEATURE_COLUMNS = [
    DIGITAL_TV_BVOD,
    INFLUENCER_MANAGEMENT,
    ONLINE_MULTIFORMAT_AMAZON,
    ONLINE_MULTIFORMAT_TESCO,
    ONLINE_VIDEO_GOOGLE,
    ONLINE_VIDEO_YOUTUBE,
    PAID_SEARCH_AWARENESS_GOOGLE,
    PAID_SEARCH_TRANSACTION_AMAZON,
    PAID_SEARCH_TRANSACTION_CITRUS,
    PAID_SEARCH_TRANSACTION_CRITEO,
    SOCIAL_MEDIA_AWARENESS_META,
    SOCIAL_MEDIA_AWARENESS_PINTEREST,
    SOCIAL_MEDIA_AWARENESS_TIKTOK,
    SOCIAL_MEDIA_TRANSACTION_META,
    SOCIAL_MEDIA_TRANSACTION_META_COLLAB,
    SOCIAL_MEDIA_TRANSACTION_TESCO,
    SOCIAL_MEDIA_TRANSACTION_THG,
    SOCIAL_MEDIA_TRANSACTION_TIKTOK,
    TESTERS_AND_MERCHANDISING,
    TRADITIONAL_TV_LINEAR,
]

# Commercial Features (control variables)
OFFLINE_AVG_PRICE = "offline_average_price"
ONLINE_AVG_PRICE = "online_average_price"
TOTAL_WEIGHTED_PROMOTION_DISTRIBUTION = "total_weighted_promotion_distribution"

CONTROL_COLUMNS = [
    OFFLINE_AVG_PRICE,
    ONLINE_AVG_PRICE,
    TOTAL_WEIGHTED_PROMOTION_DISTRIBUTION,
]

# Target
TARGET = "target"
