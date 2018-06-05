-- Create AdMaster table
-- DROP TABLE aimindreader.dbo.Comment;
DROP TABLE aimindreader.dbo.admaster;
DROP TABLE aimindreader.dbo.aiengine;

CREATE TABLE aiengine
(
	id	INT IDENTITY(1,1) PRIMARY KEY,
	rawId	INT NOT NULL,
	class1	NVARCHAR(MAX) NOT NULL,
	class2	NVARCHAR(MAX) NOT NULL,
	sentence	NVARCHAR(MAX) NOT NULL,
	sentiment	NVARCHAR(MAX) NOT NULL,
	createAt	DATE NOT NULL,
);

CREATE TABLE admaster
(
id   INT IDENTITY(1,1) PRIMARY KEY,
_projectID	NVARCHAR(MAX),
_id	NVARCHAR(MAX),
_platform	NVARCHAR(MAX) NOT NULL,
_publishedAt	NVARCHAR(MAX) NOT NULL,
_topic	NVARCHAR(MAX),
_url	NVARCHAR(MAX),
_content   NVARCHAR(MAX) NOT NULL,
_source	NVARCHAR(MAX),
_commonSentiment	NVARCHAR(MAX),
_score	NVARCHAR(MAX),
_viewCount	NVARCHAR(MAX),
_likeCount	NVARCHAR(MAX),
_commentCount	NVARCHAR(MAX),
_repostCount	NVARCHAR(MAX),
_interactCount	NVARCHAR(MAX),
_haslink	NVARCHAR(MAX),
_isOriginal	NVARCHAR(MAX),
_postFrom	NVARCHAR(MAX),
_nickName	NVARCHAR(MAX),
_uid	NVARCHAR(MAX),
_profileImageUrl	NVARCHAR(MAX),
_gender	NVARCHAR(MAX),
_province	NVARCHAR(MAX),
_city	NVARCHAR(MAX),
_description	NVARCHAR(MAX),
_friendCount	NVARCHAR(MAX),
_followerCount	NVARCHAR(MAX),
_statusCount	NVARCHAR(MAX),
_favouriteCount	NVARCHAR(MAX),
_biFollowerCount	NVARCHAR(MAX),
_createdAt	NVARCHAR(MAX),
_verified	NVARCHAR(MAX),
_verifiedType	NVARCHAR(MAX),
_verifiedReason	NVARCHAR(MAX),
_ugc	NVARCHAR(MAX),
_pbw	NVARCHAR(MAX),
_originalID	NVARCHAR(MAX),
_originalPublishedAt	NVARCHAR(MAX), 
_originalContent	NVARCHAR(MAX),
_originalSource	NVARCHAR(MAX),
_originalCommentCount	NVARCHAR(MAX),
_originalRepostCount	NVARCHAR(MAX),
_originalLikeCount	NVARCHAR(MAX),
_originalPostFrom	NVARCHAR(MAX),
_originalUID	NVARCHAR(MAX),
_originalNickName	NVARCHAR(MAX),
_originalVerified	NVARCHAR(MAX),
_isDeleted	NVARCHAR(MAX),
_spam	NVARCHAR(MAX),
_images	NVARCHAR(MAX),
_originalImages	NVARCHAR(MAX),
_rule	NVARCHAR(MAX),
_dataTag NVARCHAR(MAX),
_imageTag NVARCHAR(MAX),
_videoContent NVARCHAR(MAX),
_rewardCount	NVARCHAR(MAX),
_isReward	NVARCHAR(MAX),
_title	NVARCHAR(MAX),
_indexid	NVARCHAR(MAX),
_author	NVARCHAR(MAX),
_digest	NVARCHAR(MAX),
_userName	NVARCHAR(MAX),
_biz	NVARCHAR(MAX),
_account NVARCHAR(MAX),
_codeImageUrl NVARCHAR(MAX),
_openId	NVARCHAR(MAX),
_originalUrl	NVARCHAR(MAX),
_channel	NVARCHAR(MAX),
_secondChannel	NVARCHAR(MAX),
_thirdChannel	NVARCHAR(MAX),
_floor	NVARCHAR(MAX),
_isComment	NVARCHAR(MAX),
_isEssence	NVARCHAR(MAX),
_userUrl	NVARCHAR(MAX),
_userLevel	NVARCHAR(MAX), 
);
