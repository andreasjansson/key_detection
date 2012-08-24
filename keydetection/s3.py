import tempfile
import boto.s3.key
import boto.s3.bucket

def s3_upload(bucket_name, local_filename, s3_filename):
    conn = boto.connect_s3()
    bucket = boto.s3.bucket.Bucket(conn, bucket_name)
    key = boto.s3.key.Key(bucket)
    key.key = s3_filename
    key.set_contents_from_filename(local_filename)
    key.make_public()
    

def s3_download(bucket_name, s3_filename):
    local_file = tempfile.NamedTemporaryFile(suffix = os.path.splitext(s3_filename)[1], delete = False)
    conn = boto.connect_s3()
    bucket = boto.s3.bucket.Bucket(conn, bucket_name)
    k = boto.s3.key.Key(bucket)
    k.key = s3_filename
    k.get_contents_to_file(local_file)
    local_file.close()                                                                                                  
    return local_file.name

def s3_delete(bucket_name, s3_filename):
    conn = boto.connect_s3()
    bucket = boto.s3.bucket.Bucket(conn, bucket_name)
    try:
        bucket.delete_key(s3_filename)
    except Exception:
        logging.warning('Failed to delete ' + s3_filename)
    
